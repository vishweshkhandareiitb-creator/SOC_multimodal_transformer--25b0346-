import torch
import torch.nn as nn

class TextTransformerBlock(nn.Module):
    def __init__(self, embed_dim=192, num_heads=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class TextTransformer(nn.Module):
    def __init__(self, vocab_size=10000, max_seq_len=32, embed_dim=192, depth=4, num_heads=6, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            TextTransformerBlock(embed_dim, num_heads, 4.0, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def encode(self, text_tokens, text_mask=None):
        x = self.token_embed(text_tokens)
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.pos_drop(x)

        key_padding_mask = None
        if text_mask is not None:
            key_padding_mask = (text_mask == 0)

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
            
        x = self.norm(x)
        return x