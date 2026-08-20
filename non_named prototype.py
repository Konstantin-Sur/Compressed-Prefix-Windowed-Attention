"""
Train a hierarchical-compression language model on FineWeb-Edu.
Uses gated pooling + hierarchical attention with RoPE.

Same features as original:
- checkpoint saving/resuming
- live loss printing
- gradient accumulation
- fp16 mixed precision
- rolling tokens/sec logging
"""

import math
import time
from collections import deque
from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm

# ========== CONFIG ==========
CONFIG = {
    # --- data ---
    "dataset_name": "HuggingFaceFW/fineweb-edu",
    "dataset_subset": "sample-10BT",
    "seq_len": 512,                # must be divisible by window_size
    "batch_size": 16,

    # --- model (hierarchical) ---
    "n_layer": 6,
    "n_head": 4,
    "n_embd": 512,
    "window_size": 128,            # block size for compression
    "d_ff_mult": 4,                # SwiGLU multiplier (2/3 of 4*d_model)
    "dropout": 0.0,
    "rope_base": 10000.0,

    # --- training ---
    "max_steps": 20000,
    "lr": 3e-4,
    "min_lr_ratio": 0.1,
    "lr_decay_steps": 20000,
    "weight_decay": 0.1,
    "warmup_steps": 0,
    "grad_clip": 1.0,
    "gradient_accumulation_steps": 1,
    "eval_interval": 250,
    "eval_iters": 20,
    "print_interval": 50,
    "log_interval": 1,
    "checkpoint_interval": 1000,

    # --- precision ---
    "use_fp16": True,

    # --- misc ---
    "seed": 1337,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_path": "/content/model_hicomp.pt",
}

torch.manual_seed(CONFIG["seed"])
print(f"Using device: {CONFIG['device']}")
if CONFIG["device"] == "cpu":
    print("WARNING: no GPU detected. In Colab: Runtime > Change runtime type > GPU.")

USE_AMP = CONFIG["device"] == "cuda" and CONFIG["use_fp16"]
AMP_DTYPE = torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
print(f"fp16 mixed precision: {'ON' if USE_AMP else 'OFF'}")

# ========== TOKENIZER ==========
tokenizer = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = tokenizer.n_vocab

# ========== DATA LOADER (unchanged) ==========
class FineWebEduStream(IterableDataset):
    def __init__(self, seq_len, split="train", subset="sample-10BT"):
        super().__init__()
        self.seq_len = seq_len
        self.split = split
        self.subset = subset

    def __iter__(self):
        ds = load_dataset(
            CONFIG["dataset_name"],
            name=self.subset,
            split=self.split,
            streaming=True,
        )
        ds = ds.shuffle(seed=CONFIG["seed"], buffer_size=10_000)

        buffer = []
        eot = tokenizer.eot_token

        for example in ds:
            text = example["text"]
            if not text:
                continue
            ids = tokenizer.encode_ordinary(text)
            ids.append(eot)
            buffer.extend(ids)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

def make_loader(split, subset):
    dataset = FineWebEduStream(seq_len=CONFIG["seq_len"], split=split, subset=subset)
    return DataLoader(dataset, batch_size=CONFIG["batch_size"])

train_loader = make_loader("train", CONFIG["dataset_subset"])
val_loader = make_loader("train", CONFIG["dataset_subset"])

# ========== HIERARCHICAL MODEL (copied from your files) ==========

# ------------------------ rope.py ------------------------
def build_rope_cache(max_pos: int, head_dim: int, base: float = 10000.0, device=None):
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_pos, device=device).float()
    freqs = torch.outer(t, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin

def apply_rope(x: torch.Tensor, positions: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor):
    B, H, T, D = x.shape
    if positions.dim() == 1:
        positions = positions.unsqueeze(0).expand(B, -1)
    cos = cos_cache[positions].unsqueeze(1)   # (B,1,T,D/2)
    sin = sin_cache[positions].unsqueeze(1)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rot_x1 = x1 * cos - x2 * sin
    rot_x2 = x1 * sin + x2 * cos
    return torch.stack([rot_x1, rot_x2], dim=-1).flatten(-2)

# ------------------------ attention.py ------------------------
def build_block_causal_mask(n_compressed: int, n_raw: int, device=None) -> torch.Tensor:
    total = n_compressed + n_raw
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    if n_compressed > 0:
        comp_causal = torch.tril(torch.ones(n_compressed, n_compressed, dtype=torch.bool, device=device))
        mask[:n_compressed, :n_compressed] = comp_causal
    if n_raw > 0:
        if n_compressed > 0:
            mask[n_compressed:, :n_compressed] = True
        raw_causal = torch.tril(torch.ones(n_raw, n_raw, dtype=torch.bool, device=device))
        mask[n_compressed:, n_compressed:] = raw_causal
    return mask

class HierarchicalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def forward(self, raw, compressed, raw_positions, compressed_positions,
                cos_cache, sin_cache):
        B, n_raw, D = raw.shape
        n_comp = compressed.shape[1]
        device = raw.device

        if n_comp > 0:
            kv_src = torch.cat([compressed, raw], dim=1)
            kv_positions = torch.cat([compressed_positions, raw_positions], dim=1)
        else:
            kv_src = raw
            kv_positions = raw_positions

        k = self._split_heads(self.k_proj(kv_src))
        v = self._split_heads(self.v_proj(kv_src))
        k = apply_rope(k, kv_positions, cos_cache, sin_cache)

        new_compressed_out = None
        if n_comp > 0:
            q_comp = self._split_heads(self.q_proj(compressed))
            q_comp = apply_rope(q_comp, compressed_positions, cos_cache, sin_cache)
            k_comp = k[:, :, :n_comp, :]
            v_comp = v[:, :, :n_comp, :]
            comp_mask = build_block_causal_mask(n_comp, 0, device=device)[:n_comp, :n_comp]
            attn_mask = comp_mask.unsqueeze(0).unsqueeze(0)
            out_comp = F.scaled_dot_product_attention(
                q_comp, k_comp, v_comp, attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
            new_compressed_out = self.out_proj(self._merge_heads(out_comp))

        q_raw = self._split_heads(self.q_proj(raw))
        q_raw = apply_rope(q_raw, raw_positions, cos_cache, sin_cache)

        full_mask = build_block_causal_mask(n_comp, n_raw, device=device)
        raw_rows_mask = full_mask[n_comp:, :]
        attn_mask = raw_rows_mask.unsqueeze(0).unsqueeze(0)

        out_raw = F.scaled_dot_product_attention(
            q_raw, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        new_raw_out = self.out_proj(self._merge_heads(out_raw))

        return new_raw_out, new_compressed_out

# ------------------------ compressor.py ------------------------
class GatedPooler(nn.Module):
    def __init__(self, d_model: int, block_size: int, gate_hidden_mult: int = 1, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
        self.in_block_pos_emb = nn.Embedding(block_size, d_model)
        if gate_hidden_mult and gate_hidden_mult > 0:
            hidden = d_model * gate_hidden_mult
            self.gate = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_model),
            )
        else:
            self.gate = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block_size = x.shape[-2]
        assert block_size == self.block_size, f"block_size mismatch: {block_size} vs {self.block_size}"
        pos_ids = torch.arange(block_size, device=x.device)
        pos_emb = self.in_block_pos_emb(pos_ids)
        gate_input = x + pos_emb
        gate_logits = self.gate(gate_input)
        weights = F.softmax(gate_logits, dim=-2)
        weights = self.dropout(weights)
        values = self.value_proj(x)
        compressed = (weights * values).sum(dim=-2)
        return self.out_norm(compressed)

# ------------------------ model.py (adapted) ------------------------
@dataclass
class HConfig:
    vocab_size: int
    seq_len: int                # total context length
    n_layer: int
    n_head: int
    n_embd: int
    window_size: int
    d_ff_mult: int = 4
    dropout: float = 0.0
    rope_base: float = 10000.0

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: int = 4):
        super().__init__()
        hidden = int(d_model * mult * 2 / 3)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class HLayer(nn.Module):
    def __init__(self, cfg: HConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = HierarchicalAttention(cfg.n_embd, cfg.n_head, dropout=cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ffn = SwiGLU(cfg.n_embd, cfg.d_ff_mult)
        self.ln_pool = nn.LayerNorm(cfg.n_embd)
        self.pooler = GatedPooler(cfg.n_embd, cfg.window_size)

    def forward_training(self, h: torch.Tensor):
        B, T, D = h.shape
        bs = self.cfg.window_size
        device = h.device

        n_blocks = T // bs
        tail_len = T % bs

        h_norm = self.ln1(h)
        cos_cache, sin_cache = self._rope_cache(T, device)

        if n_blocks > 0:
            h_blocks = h_norm[:, : n_blocks * bs, :].view(B, n_blocks, bs, D)
            compressed_all = self.ln_pool(self.pooler(h_blocks))
            comp_positions_all = (torch.arange(n_blocks, device=device) + 1) * bs - 1
            comp_positions_all = comp_positions_all.unsqueeze(0).expand(B, -1)
        else:
            compressed_all = h_norm.new_zeros(B, 0, D)
            comp_positions_all = torch.zeros(B, 0, dtype=torch.long, device=device)

        attn_out_chunks = []
        for i in range(n_blocks):
            raw_chunk = h_norm[:, i * bs:(i + 1) * bs, :]
            raw_pos = torch.arange(i * bs, (i + 1) * bs, device=device).unsqueeze(0).expand(B, -1)
            comp_prefix = compressed_all[:, :i, :]
            comp_prefix_pos = comp_positions_all[:, :i]
            raw_out, _ = self.attn(
                raw=raw_chunk,
                compressed=comp_prefix,
                raw_positions=raw_pos,
                compressed_positions=comp_prefix_pos,
                cos_cache=cos_cache, sin_cache=sin_cache,
            )
            attn_out_chunks.append(raw_out)

        if tail_len > 0:
            raw_chunk = h_norm[:, n_blocks * bs:, :]
            raw_pos = torch.arange(n_blocks * bs, T, device=device).unsqueeze(0).expand(B, -1)
            raw_out, _ = self.attn(
                raw=raw_chunk,
                compressed=compressed_all,
                raw_positions=raw_pos,
                compressed_positions=comp_positions_all,
                cos_cache=cos_cache, sin_cache=sin_cache,
            )
            attn_out_chunks.append(raw_out)

        attn_out = torch.cat(attn_out_chunks, dim=1)
        h = h + attn_out
        h = h + self.ffn(self.ln2(h))
        return h

    def _rope_cache(self, T, device):
        max_pos = max(T, self.cfg.window_size) + 8
        head_dim = self.cfg.n_embd // self.cfg.n_head
        return build_rope_cache(max_pos, head_dim, base=self.cfg.rope_base, device=device)

class HierarchicalModel(nn.Module):
    def __init__(self, cfg: HConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.n_embd)   # still used for added positional info? Not used in HLayer, but kept for compatibility? Actually we use RoPE, so we don't need learned pos emb. But we can keep it for simplicity, though it will be unused. We'll remove it.
        # Actually our model uses RoPE, not learned pos embeddings. Remove pos_emb.
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([HLayer(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying optional – we'll keep untied to match original script's expectation of separate lm_head.
        # But we can tie if desired; not doing so adds parameters.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.seq_len
        # No learned pos emb; RoPE is applied inside HLayer.
        x = self.drop(self.tok_emb(idx))   # (B, T, D)
        for layer in self.layers:
            x = layer.forward_training(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.seq_len else idx[:, -self.cfg.seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# ========== BUILD MODEL ==========
hcfg = HConfig(
    vocab_size=VOCAB_SIZE,
    seq_len=CONFIG["seq_len"],
    n_layer=CONFIG["n_layer"],
    n_head=CONFIG["n_head"],
    n_embd=CONFIG["n_embd"],
    window_size=CONFIG["window_size"],
    d_ff_mult=CONFIG["d_ff_mult"],
    dropout=CONFIG["dropout"],
    rope_base=CONFIG["rope_base"],
)
model = HierarchicalModel(hcfg).to(CONFIG["device"])
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params / 1e6:.2f}M")
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_trainable / 1e6:.2f}M")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"],
    betas=(0.9, 0.95),
)

def lr_at_step(step):
    if step < CONFIG["warmup_steps"]:
        return CONFIG["lr"] * (step + 1) / CONFIG["warmup_steps"]
    if step >= CONFIG["lr_decay_steps"]:
        return CONFIG["lr"] * CONFIG["min_lr_ratio"]
    progress = (step - CONFIG["warmup_steps"]) / max(1, CONFIG["lr_decay_steps"] - CONFIG["warmup_steps"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_ratio = CONFIG["min_lr_ratio"]
    return CONFIG["lr"] * (min_ratio + (1 - min_ratio) * coeff)

# ========== VALIDATION LOSS ==========
@torch.no_grad()
def estimate_val_loss(val_iter):
    model.eval()
    losses = []
    for _ in range(CONFIG["eval_iters"]):
        x, y = next(val_iter)
        x, y = x.to(CONFIG["device"]), y.to(CONFIG["device"])
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

# ========== CHECKPOINT HELPERS ==========
def save_checkpoint(step, train_steps, train_losses, val_steps, val_losses):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": step,
        "train_steps": train_steps,
        "train_losses": train_losses,
        "val_steps": val_steps,
        "val_losses": val_losses,
        "config": CONFIG,
    }
    torch.save(checkpoint, CONFIG["checkpoint_path"])
    print(f"Checkpoint saved to {CONFIG['checkpoint_path']} at step {step}")

def load_checkpoint(path):
    ckpt = torch.load(path, map_location=CONFIG["device"])
    model.load_state_dict(ckpt["model_state_dict"])
    # optimizer.load_state_dict(ckpt["optimizer_state_dict"])  # uncomment if needed
    if USE_AMP and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_step = ckpt["step"]
    train_steps = ckpt.get("train_steps", [])
    train_losses = ckpt.get("train_losses", [])
    val_steps = ckpt.get("val_steps", [])
    val_losses = ckpt.get("val_losses", [])
    print(f"Resumed from checkpoint at step {start_step}")
    return start_step, train_steps, train_losses, val_steps, val_losses

# ========== TRAINING LOOP ==========
def train(resume_from=None):
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    if resume_from is not None:
        start_step, train_steps, train_losses, val_steps, val_losses = load_checkpoint(resume_from)
    else:
        start_step = 0
        train_steps, train_losses = [], []
        val_steps, val_losses = [], []

    acc_steps = CONFIG["gradient_accumulation_steps"]
    effective_batch_size = CONFIG["batch_size"] * acc_steps
    tokens_per_optim_step = effective_batch_size * CONFIG["seq_len"]
    print_interval = CONFIG["print_interval"]

    model.train()
    pbar = tqdm(range(start_step, CONFIG["max_steps"]), desc="training")

    t0 = time.time()
    recent_step_times = deque(maxlen=20)
    step_t0 = time.time()

    for step in pbar:
        lr = lr_at_step(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        running_loss = 0.0
        for micro_step in range(acc_steps):
            x, y = next(train_iter)
            x, y = x.to(CONFIG["device"]), y.to(CONFIG["device"])

            with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
                _, loss = model(x, y)
                loss = loss / acc_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * acc_steps

        if USE_AMP:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])

        if USE_AMP:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step_dt = time.time() - step_t0
        recent_step_times.append(step_dt)
        step_t0 = time.time()

        recent_tok_s = tokens_per_optim_step / (sum(recent_step_times) / len(recent_step_times))
        cumulative_elapsed = max(1e-6, time.time() - t0)
        cumulative_steps_this_session = step - start_step + 1
        avg_tok_s = (cumulative_steps_this_session * tokens_per_optim_step) / cumulative_elapsed

        train_steps.append(step)
        train_losses.append(running_loss / acc_steps)

        if step % CONFIG["log_interval"] == 0 or step == start_step:
            pbar.set_postfix(
                loss=f"{running_loss/acc_steps:.3f}",
                lr=f"{lr:.2e}",
                tok_s=f"{recent_tok_s:,.0f}",
                avg_tok_s=f"{avg_tok_s:,.0f}",
            )

        if (step + 1) % print_interval == 0 or step == CONFIG["max_steps"] - 1:
            display_step = (step + 1) // print_interval
            print(
                f"step {display_step} loss:{train_losses[-1]:.3f} "
                f"(actual step {step+1}) | {recent_tok_s:,.0f} tok/s "
                f"(avg {avg_tok_s:,.0f} tok/s)"
            )

        if step % CONFIG["eval_interval"] == 0 or step == CONFIG["max_steps"] - 1:
            v_loss = estimate_val_loss(val_iter)
            val_steps.append(step)
            val_losses.append(v_loss)
            print(f"Validation loss at step {step+1}: {v_loss:.3f}")

        if step % CONFIG["checkpoint_interval"] == 0 and step > start_step:
            save_checkpoint(step, train_steps, train_losses, val_steps, val_losses)

    save_checkpoint(CONFIG["max_steps"], train_steps, train_losses, val_steps, val_losses)
    print(f"Training finished. Final model saved to {CONFIG['checkpoint_path']}")
    return train_steps, train_losses, val_steps, val_losses

# ========== GENERATION SAMPLE ==========
def sample(prompt="The history of ", max_new_tokens=100):
    model.eval()
    ids = tokenizer.encode_ordinary(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=CONFIG["device"])
    out = model.generate(idx, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(out[0].tolist())
    model.train()
    return text

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    # train(resume_from="/content/model_hicomp.pt")  # to resume
    train()
    print("\n--- sample generation ---")
    print(sample())
