import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, init_temperature=0.07):
        super().__init__()
        self.log_inv_tau = nn.Parameter(
            torch.tensor([1.0 / init_temperature]).log()
        )

    def forward(self, image_embeds, text_embeds):
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        log_inv_tau = self.log_inv_tau.clamp(0, 4.6052)
        inv_tau = log_inv_tau.exp()

        logits = inv_tau * image_embeds @ text_embeds.T

        N = image_embeds.size(0)
        labels = torch.arange(N, device=logits.device)

        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2
