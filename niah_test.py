import math, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
random.seed(0)

# ------------------------------------------------------------------
# CONFIG -- edit these
# ------------------------------------------------------------------
VOCAB_SIZE   = 64        # tiny synthetic vocab: '#' filler + digits + special tokens
D_MODEL      = 128
N_HEADS      = 4
N_LAYERS     = 4
WINDOW_SIZE  = 128        # CPWA block/window size
DROPOUT      = 0.0

TRAIN_LEN        = 2048   # sequence length used for training both models
BATCH_SIZE       = 16
TRAIN_STEPS      = 1500
LR               = 3e-4

# Evaluate both models on every length (tiny model → usually fits)
EVAL_LENGTHS     = [512, 1024, 2048, 4096, 8192, 16384]
EVAL_SAMPLES_PER_LEN = 100

# ------------------------------------------------------------------
# Synthetic NIAH data (RULER-style): haystack = repeated '#', one needle.
# Sequence layout:  [ '#' ... #  NEEDLE_TOK  VALUE_TOK  #...#  QUERY_TOK ]
# Target: predict VALUE_TOK at the final position, conditioned on QUERY_TOK.
# Token ids: 0='#', 1..10 = values (digits), 11 = NEEDLE marker, 12 = QUERY marker
# ------------------------------------------------------------------
HASH_TOK   = 0
NEEDLE_TOK = 11
QUERY_TOK  = 12
VALUE_LO, VALUE_HI = 1, 10   # inclusive value token range

def make_batch(batch_size, seq_len, device):
    """
    Returns:
      x: (B, T) input ids -- full sequence including the query token at the end
      y: (B,)   target value id (the needle's value) to be predicted at last position
    Needle placed at a random position in [1, T-3] so there's room for
    NEEDLE_TOK, VALUE_TOK pair plus the trailing QUERY_TOK.
    """
    x = torch.full((batch_size, seq_len), HASH_TOK, dtype=torch.long)
    y = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        value = random.randint(VALUE_LO, VALUE_HI)
        needle_pos = random.randint(1, seq_len - 3)
        x[b, needle_pos] = NEEDLE_TOK
        x[b, needle_pos + 1] = value
        x[b, -1] = QUERY_TOK
        y[b] = value
    return x.to(device), y.to(device)


def build_rope_cache(max_pos, head_dim, base=10000.0, device=None):
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_pos, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, positions, cos_cache, sin_cache):
    B, H, T, D = x.shape
    if positions.dim() == 1:
        positions = positions.unsqueeze(0).expand(B, -1)
    cos = cos_cache[positions].unsqueeze(1)
    sin = sin_cache[positions].unsqueeze(1)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rot_x1 = x1 * cos - x2 * sin
    rot_x2 = x1 * sin + x2 * cos
    return torch.stack([rot_x1, rot_x2], dim=-1).flatten(-2)

def build_block_causal_mask(n_compressed, n_raw, device=None):
    total = n_compressed + n_raw
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    if n_compressed > 0:
        mask[:n_compressed, :n_compressed] = torch.tril(torch.ones(n_compressed, n_compressed, dtype=torch.bool, device=device))
    if n_raw > 0:
        if n_compressed > 0:
            mask[n_compressed:, :n_compressed] = True
        mask[n_compressed:, n_compressed:] = torch.tril(torch.ones(n_raw, n_raw, dtype=torch.bool, device=device))
    return mask


class GatedPooler(nn.Module):
    def __init__(self, d_model, block_size, gate_hidden_mult=1, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
        self.in_block_pos_emb = nn.Embedding(block_size, d_model)
        if gate_hidden_mult and gate_hidden_mult > 0:
            hidden = d_model * gate_hidden_mult
            self.gate = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        else:
            self.gate = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        block_size = x.shape[-2]
        assert block_size == self.block_size, f"block_size mismatch: {block_size} vs {self.block_size}"
        pos_ids = torch.arange(block_size, device=x.device)
        pos_emb = self.in_block_pos_emb(pos_ids)
        gate_logits = self.gate(x + pos_emb)
        weights = self.dropout(F.softmax(gate_logits, dim=-2))
        values = self.value_proj(x)
        compressed = (weights * values).sum(dim=-2)
        return self.out_norm(compressed)


class CPWAAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size, rope_base=10000.0, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.rope_base = rope_base
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.ln_pool = nn.LayerNorm(d_model)
        self.pooler = GatedPooler(d_model, window_size)

    def _split_heads(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def _rope_cache(self, T, device):
        max_pos = max(T, self.window_size) + 8
        return build_rope_cache(max_pos, self.head_dim, base=self.rope_base, device=device)

    def _attend(self, raw, compressed, raw_positions, compressed_positions, cos_cache, sin_cache):
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

        q_raw = self._split_heads(self.q_proj(raw))
        q_raw = apply_rope(q_raw, raw_positions, cos_cache, sin_cache)

        full_mask = build_block_causal_mask(n_comp, n_raw, device=device)
        attn_mask = full_mask[n_comp:, :].unsqueeze(0).unsqueeze(0)

        out_raw = F.scaled_dot_product_attention(
            q_raw, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(self._merge_heads(out_raw))

    def forward(self, h_norm):
        B, T, D = h_norm.shape
        bs = self.window_size
        device = h_norm.device

        n_blocks = T // bs
        tail_len = T % bs
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
            attn_out_chunks.append(self._attend(
                raw=raw_chunk, compressed=compressed_all[:, :i, :],
                raw_positions=raw_pos, compressed_positions=comp_positions_all[:, :i],
                cos_cache=cos_cache, sin_cache=sin_cache,
            ))

        if tail_len > 0:
            raw_chunk = h_norm[:, n_blocks * bs:, :]
            raw_pos = torch.arange(n_blocks * bs, T, device=device).unsqueeze(0).expand(B, -1)
            attn_out_chunks.append(self._attend(
                raw=raw_chunk, compressed=compressed_all,
                raw_positions=raw_pos, compressed_positions=comp_positions_all,
                cos_cache=cos_cache, sin_cache=sin_cache,
            ))

        return torch.cat(attn_out_chunks, dim=1)

# ------------------------------------------------------------------
# Plain full (dense causal) attention -- baseline for comparison
# ------------------------------------------------------------------
class FullAttention(nn.Module):
    def __init__(self, d_model, n_heads, rope_base=10000.0, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope_base = rope_base
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x):
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def forward(self, h_norm):
        B, T, D = h_norm.shape
        device = h_norm.device
        cos_cache, sin_cache = build_rope_cache(T + 8, self.head_dim, base=self.rope_base, device=device)
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        q = apply_rope(self._split_heads(self.q_proj(h_norm)), positions, cos_cache, sin_cache)
        k = apply_rope(self._split_heads(self.k_proj(h_norm)), positions, cos_cache, sin_cache)
        v = self._split_heads(self.v_proj(h_norm))

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(self._merge_heads(out))

# ------------------------------------------------------------------
# Transformer block + tiny GPT wrapper (attention type is pluggable)
# ------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d_model, n_heads, attn_type, window_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        if attn_type == "cpwa":
            self.attn = CPWAAttention(d_model, n_heads, window_size, dropout=dropout)
        elif attn_type == "full":
            self.attn = FullAttention(d_model, n_heads, dropout=dropout)
        else:
            raise ValueError(attn_type)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, attn_type, window_size, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, attn_type, window_size, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.tok_emb(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.head(h)  # (B, T, vocab)

# ------------------------------------------------------------------
# Train / eval loops
# ------------------------------------------------------------------
def train_model(attn_type, seq_len, steps, batch_size, log_every=100):
    model = TinyGPT(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, attn_type, WINDOW_SIZE, DROPOUT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    t0 = time.time()
    loss_history = []
    for step in range(1, steps + 1):
        x, y = make_batch(batch_size, seq_len, device)
        logits = model(x)              # (B, T, vocab)
        last_logits = logits[:, -1, :]  # predict needle value at final (query) position
        loss = F.cross_entropy(last_logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_history.append(loss.item())
        if step % log_every == 0 or step == 1:
            print(f"  [{attn_type}] step {step:4d}/{steps} loss {loss.item():.4f} ({time.time()-t0:.1f}s)")
    return model, loss_history

@torch.no_grad()
def eval_niah_accuracy(model, seq_len, n_samples, batch_size=16):
    model.eval()
    correct, total = 0, 0
    remaining = n_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        x, y = make_batch(bs, seq_len, device)
        logits = model(x)[:, -1, :]
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += bs
        remaining -= bs
    return 100.0 * correct / total

# ------------------------------------------------------------------
# Plotting -- clean, scientific-paper style figures
# ------------------------------------------------------------------
COLOR_FULL = "#1f6feb"   # blue
COLOR_CPWA = "#e8590c"   # orange
COLOR_GRID = "#d0d0d0"

def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, color=COLOR_GRID, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#333333", labelsize=10)

def plot_results(loss_full, loss_cpwa, lengths, acc_full, acc_cpwa, train_len, save_path="niah_results.png"):
    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["axes.titleweight"] = "bold"
    mpl.rcParams["axes.edgecolor"] = "#444444"

    fig = plt.figure(figsize=(13, 9), dpi=150)
    fig.suptitle("Synthetic NIAH: Full Attention vs. CPWA (trained from scratch)",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.35, wspace=0.28)

    # ---- (1) Training loss curves ----
    ax1 = fig.add_subplot(gs[0, 0])
    def smooth(y, k=15):
        if len(y) < k:
            return y
        kernel = torch.ones(k) / k
        y_t = torch.tensor(y, dtype=torch.float32)
        padded = F.pad(y_t.view(1, 1, -1), (k // 2, k - 1 - k // 2), mode="replicate")
        return F.conv1d(padded, kernel.view(1, 1, -1)).view(-1).tolist()

    ax1.plot(smooth(loss_full), color=COLOR_FULL, linewidth=1.8, label="Full attention")
    ax1.plot(smooth(loss_cpwa), color=COLOR_CPWA, linewidth=1.8, label="CPWA")
    ax1.set_title(f"Training loss  (seq_len={train_len})", fontsize=11)
    ax1.set_xlabel("Step", fontsize=10)
    ax1.set_ylabel("Cross-entropy loss", fontsize=10)
    ax1.legend(frameon=False, fontsize=9, loc="upper right")
    _style_axis(ax1)

    # ---- (2) Zoomed final-loss bar (last 10% of training) ----
    ax2 = fig.add_subplot(gs[0, 1])
    tail = max(1, len(loss_full) // 10)
    final_full = sum(loss_full[-tail:]) / tail
    final_cpwa = sum(loss_cpwa[-tail:]) / tail
    bars = ax2.bar(["Full", "CPWA"], [final_full, final_cpwa],
                   color=[COLOR_FULL, COLOR_CPWA], width=0.5, edgecolor="white", linewidth=1.2)
    for b, v in zip(bars, [final_full, final_cpwa]):
        ax2.text(b.get_x() + b.get_width() / 2, v + max(final_full, final_cpwa) * 0.02,
                 f"{v:.3f}", ha="center", fontsize=10, fontweight="bold", color="#222222")
    ax2.set_title(f"Final loss (avg of last {tail} steps)", fontsize=11)
    ax2.set_ylabel("Cross-entropy loss", fontsize=10)
    _style_axis(ax2)
    ax2.spines["bottom"].set_visible(True)

    # ---- (3) Accuracy vs context length -- the main scientific plot ----
    ax3 = fig.add_subplot(gs[1, :])

    ax3.plot(lengths, acc_full, marker="o", markersize=9, linewidth=3.0,
             color=COLOR_FULL, label="Full attention (dense)", zorder=4,
             markeredgecolor="white", markeredgewidth=1.2)
    ax3.plot(lengths, acc_cpwa, marker="s", markersize=7, linewidth=2.2,
             color=COLOR_CPWA, label="CPWA", zorder=3, linestyle="--",
             markeredgecolor="white", markeredgewidth=1.0)

    # Vertical line + clean label for the training length
    ax3.axvline(train_len, color="#2ca02c", linestyle=":", linewidth=1.8, zorder=1)
    ax3.text(train_len, 102, "train length", fontsize=9, color="#2ca02c",
             ha="center", va="bottom", fontweight="bold")

    # Explanation that everything after train_len is pure length generalisation
    ax3.text(train_len * 1.35, 12,
             "← models trained at 2048\n   everything to the right\n   is length generalisation",
             fontsize=8.5, color="#555555", va="bottom", ha="left")

    ax3.set_xscale("log", base=2)
    ax3.set_xticks(lengths)
    ax3.set_xticklabels([str(l) for l in lengths], fontsize=9.5)
    ax3.set_ylim(-3, 108)
    ax3.set_xlabel("Context length (tokens, log scale)", fontsize=11)
    ax3.set_ylabel("NIAH retrieval accuracy (%)", fontsize=11)
    ax3.set_title("Needle-in-a-Haystack accuracy vs. context length", fontsize=12)
    ax3.legend(frameon=False, fontsize=10, loc="lower left")
    _style_axis(ax3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved figure to: {save_path}")
    plt.show()

# ========================================================================
# RUN
# ========================================================================
if __name__ == "__main__":
    print(f"device = {device}\n")

    # ---- Train both models at TRAIN_LEN ----
    print(f"=== Training FULL attention model @ seq_len={TRAIN_LEN} ===")
    full_model, loss_full = train_model("full", TRAIN_LEN, TRAIN_STEPS, BATCH_SIZE)

    print(f"\n=== Training CPWA model @ seq_len={TRAIN_LEN} ===")
    cpwa_model, loss_cpwa = train_model("cpwa", TRAIN_LEN, TRAIN_STEPS, BATCH_SIZE)

    # ---- Evaluate both models on every length ----
    print("\n=== Comparison: CPWA vs Full attention (all lengths) ===")
    print(f"{'seq_len':>8} | {'Full acc%':>10} | {'CPWA acc%':>10}")
    acc_full, acc_cpwa = [], []
    for L in EVAL_LENGTHS:
        full_acc = eval_niah_accuracy(full_model, L, EVAL_SAMPLES_PER_LEN)
        cpwa_acc = eval_niah_accuracy(cpwa_model, L, EVAL_SAMPLES_PER_LEN)
        acc_full.append(full_acc)
        acc_cpwa.append(cpwa_acc)
        print(f"{L:>8} | {full_acc:>9.1f}% | {cpwa_acc:>9.1f}%")

    # ---- Plot everything ----
    plot_results(
        loss_full=loss_full,
        loss_cpwa=loss_cpwa,
        lengths=EVAL_LENGTHS,
        acc_full=acc_full,
        acc_cpwa=acc_cpwa,
        train_len=TRAIN_LEN,
    )

    print("\nDone.")
