# Compressed‑Prefix Window Attention (CPWA)

**CPWA** is a novel attention layer that enables efficient processing of long sequences by compressing completed blocks of tokens into compact summaries, while preserving causal attention over the current sliding window.

---

## Overview

Given a sequence of tokens, CPWA divides it into non‑overlapping blocks of fixed size `B` (e.g., 128). For each *completed* block, a **gated pooler** compresses it into a single vector. At any point, the *live* context consists of:

- A **compressed prefix** – one vector per completed block, summarising all tokens in that block.
- A **raw window** – the current incomplete block of up to `B` raw tokens.

The layer performs attention where:
- **Raw tokens** attend to **all compressed vectors** from the past **and** causally to each other in the current window.
- **Compressed tokens** attend **causally only among themselves** – they never see raw tokens.

This yields **O(T·B)** training complexity (instead of O(T²)) and **O(T/B)** memory for the compressed history, while retaining full access to the entire past.

---

## Key Components

### 1. Gated Pooler (block compressor)
For each block of `B` token embeddings `X ∈ ℝ^(B×D)`, it computes:

- **Gate logits**: `G = gate(X + pos_emb)`  
  (positional embedding inside the block)
- **Softmax weights** (per‑channel):  
  `W = softmax(G, dim=0)`   – each feature dimension learns its own attention over positions.
- **Values**: `V = value_proj(X)`
- **Compressed vector**: `c = Σ (W ⊙ V)`, then `LayerNorm`

This is *not* simple averaging or max pooling – it learns which positions in the block are most informative per channel.

### 2. Hierarchical Attention Pattern
The combined sequence is arranged as:

```
[compressed_0, compressed_1, …, compressed_k, raw_0, raw_1, …, raw_{m-1}]
```

where `k` = number of completed blocks, `m` = number of tokens in current window.

The attention mask is:

| Query \ Key | compressed 0…k | raw 0…m-1 |
|-------------|----------------|-----------|
| compressed_i | causal among compressed (i→j≤i) | ❌ no access |
| raw_j       | ✅ full access to all compressed | causal among raw (j→l≤j) |

**Intuition**: Compressed vectors represent the *past* and are fixed; raw tokens see the entire past at low cost, while compressed tokens stay independent of future raw content (cache‑stable for generation).

### 3. Rotary Position Embeddings (RoPE)
- **Raw tokens** get RoPE at their true global position.
- **Compressed token `i`** gets RoPE at the **end position** of its block (`(i+1)*B - 1`), preserving relative‑distance semantics.

---

## Results:

| training mode | loss |
|-------------|---------|
| pre-training | 3.5 |
| fine tune | 3.0 |

**Generation example**

Prompt:
```
####Human####:
What's the most popular programming language?
```

Result:
```
####Human####:
What's the most popular programming language?

####Assistant####: 
In terms of coding, there are several popular programming languages that can be used to create a user experience. Some popular programming languages include Python, JavaScript, Java, and JavaScript.
```

**LONGER GENERATIONS SEE IN Gen_examples.txt**

---

## Benefits

| Feature | Benefit |
|---------|---------|
| **Linear‑ish complexity** | O(T·B) instead of O(T²) – scales to long contexts. |
| **Full past access** | Raw tokens see the entire history via compressed summaries. |
| **Low memory** | Stores only `T/B` compressed vectors + one window. |
| **No position‑emb table** | RoPE handles positions without extra parameters. |
| **Fast training** | Chunked attention fits in small VRAM and achieves high token/s. |
| **Simple inference** | Cache compressed tokens as they are finalised; no recomputation. |

---

## Code Integration

The layer is implemented in PyTorch as:

- `CPWAAttention` – the hierarchical attention mechanism.
- `GatedPooler` – the block compressor.
- `CPWA` – a full transformer block combining pre‑norm, CPWA attention, and SwiGLU FFN.
- `CPWAModel` – the full language model with embedding, stack of CPWA layers, and output head.

**Example config:**
```python
cfg = CPWAConfig(
    vocab_size=50257,
    seq_len=2048,
    n_layer=6,
    n_head=4,
    n_embd=512,
    window_size=128,
    d_ff_mult=4,
)
model = CPWAModel(cfg)
```

---

## References

- Rotary Position Embedding (RoPE) – Su et al.
- Sliding window attention (used in Longformer, Mistral).
- Gated pooling – inspired by attention pooling, but per‑channel.

**CPWA combines these ideas into a single, efficient layer that learns to summarise the past while preserving causal attention on the present.**
