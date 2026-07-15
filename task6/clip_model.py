import torch
import torch.nn as nn
from loss import InfoNCELoss

class CLIPStyleModel(nn.Module):
    def __init__(self, vit_encoder, text_encoder, embed_dim=192, projection_dim=128):
        super().__init__()
        self.vit = vit_encoder
        self.text = text_encoder
        self.image_proj = nn.Linear(embed_dim, projection_dim, bias=False)
        self.text_proj = nn.Linear(embed_dim, projection_dim, bias=False)
        self.loss_fn = InfoNCELoss(init_temperature=0.07)

    def encode_image(self, images):
        feats = self.vit.encode(images)
        cls = feats[:, 0]
        return self.image_proj(cls)

    def encode_text(self, text_tokens, text_mask=None):
        feats = self.text.encode(text_tokens)
        if text_mask is not None:
            mask = text_mask.unsqueeze(-1).float()
            pooled = (feats * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        else:
            pooled = feats.mean(1)
        return self.text_proj(pooled)

    def forward(self, images, text_tokens, text_mask=None):
        img_e = self.encode_image(images)
        txt_e = self.encode_text(text_tokens, text_mask)
        loss = self.loss_fn(img_e, txt_e)
        return loss, img_e, txt_e
