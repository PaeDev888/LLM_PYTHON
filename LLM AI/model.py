"""
LLM (Large Language Model) แบบ Transformer ตั้งแต่ต้น
สร้างด้วย Python และ PyTorch เท่านั้น
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────
# 1. Token Embedding + Positional Encoding
# ─────────────────────────────────────────────

class TokenEmbedding(nn.Module):
    """แปลง token ID เป็น dense vector"""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x: (batch, seq_len)  →  (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (Vaswani et al. 2017)
    บอกโมเดลว่า token แต่ละตัวอยู่ตำแหน่งไหนใน sequence
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # สร้าง positional encoding matrix (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # ตำแหน่งคู่ → sin
        pe[:, 1::2] = torch.cos(position * div_term)  # ตำแหน่งคี่ → cos
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────
# 2. Multi-Head Scaled Dot-Product Attention
# ─────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

    แบ่งออกเป็น h หัว (heads) แต่ละหัวเรียนรู้
    ความสัมพันธ์คนละมุมมอง แล้วนำมา concat กัน
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model ต้องหาร n_heads ลงตัว"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # ขนาด vector ต่อหัว

        # Linear projection สำหรับ Q, K, V และ output
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        """(batch, seq, d_model) → (batch, heads, seq, d_k)"""
        x = x.view(batch_size, -1, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project ไปเป็น Q, K, V แล้วแบ่ง heads
        Q = self.split_heads(self.W_q(x), batch_size)  # (B, H, S, d_k)
        K = self.split_heads(self.W_k(x), batch_size)
        V = self.split_heads(self.W_v(x), batch_size)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Causal mask: ป้องกันไม่ให้มองอนาคต (สำหรับ autoregressive LM)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum ของ Values
        context = torch.matmul(attn_weights, V)  # (B, H, S, d_k)

        # Concat heads กลับมา
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)

        return self.W_o(context)


# ─────────────────────────────────────────────
# 3. Feed-Forward Network
# ─────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
    ขยาย dimension ขึ้น 4x แล้วบีบกลับ
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),          # GELU ทันสมัยกว่า ReLU สำหรับ LLM
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# 4. Transformer Block (1 layer)
# ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    1 layer ของ Transformer:
      x → LayerNorm → Attention → +residual
        → LayerNorm → FFN      → +residual
    (Pre-LayerNorm style ตาม GPT-2/3)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention + residual
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        # FFN + residual
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


# ─────────────────────────────────────────────
# 5. LLM หลัก (Decoder-only เหมือน GPT)
# ─────────────────────────────────────────────

class MiniLLM(nn.Module):
    """
    โมเดลภาษาขนาดเล็ก (Decoder-only Transformer)
    สถาปัตยกรรมเหมือน GPT แต่ขนาดเล็กกว่า

    Args:
        vocab_size : จำนวน token ทั้งหมด
        d_model    : ขนาด embedding dimension
        n_heads    : จำนวน attention heads
        n_layers   : จำนวน Transformer layers
        d_ff       : ขนาด hidden ใน FFN (โดยปกติ 4×d_model)
        max_len    : ความยาว sequence สูงสุด
        dropout    : dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Embedding layers
        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # Stack of Transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Final layer norm + output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: ใช้ embedding weight เดียวกับ output head (ประหยัด param)
        self.head.weight = self.token_emb.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)
        print(f"MiniLLM พร้อมใช้งาน | จำนวนพารามิเตอร์: {self.count_params():,}")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def make_causal_mask(self, seq_len, device):
        """สร้าง causal mask: token ที่ i มองได้เฉพาะ 0..i (ไม่มองอนาคต)"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids : (batch, seq_len) — token IDs
            targets   : (batch, seq_len) — shifted input สำหรับ training

        Returns:
            logits : (batch, seq_len, vocab_size)
            loss   : cross-entropy loss (ถ้ามี targets)
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Causal mask
        mask = self.make_causal_mask(S, device)

        # Embedding + positional encoding
        x = self.token_emb(input_ids)   # (B, S, d_model)
        x = self.pos_enc(x)

        # ผ่าน Transformer layers ทีละ layer
        for layer in self.layers:
            x = layer(x, mask)

        # Final norm + project to vocab
        x = self.ln_f(x)
        logits = self.head(x)  # (B, S, vocab_size)

        # คำนวณ loss ถ้ามี targets
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        สร้างข้อความแบบ autoregressive

        Args:
            input_ids      : (1, seq_len) — prompt
            max_new_tokens : จำนวน token ที่จะสร้างเพิ่ม
            temperature    : ยิ่งต่ำยิ่ง deterministic, ยิ่งสูงยิ่ง creative
            top_k          : เลือกจาก k token ที่น่าจะเป็นที่สุด
            top_p          : nucleus sampling — cumulative probability
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # ตัด context ถ้ายาวเกิน max_len
            ctx = generated[:, -self.max_len:]

            # Forward pass
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / temperature  # เอาเฉพาะตำแหน่งสุดท้าย

            # Top-k filtering
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits[logits < values[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprob = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumprob - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated
