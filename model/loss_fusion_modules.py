import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLossFusion(nn.Module):
    """Fuse classification and affective losses without changing the backbone outputs."""

    def __init__(
        self,
        aff_loss_type="MSE",
        aff_loss_ramp_epochs=10,
        loss_min_cls_weight=0.6,
        loss_min_aff_weight=0.05,
        loss_entropy_weight=0.01,
    ):
        super().__init__()
        self.aff_loss_type = str(aff_loss_type)
        self.aff_loss_ramp_epochs = int(aff_loss_ramp_epochs)
        self.loss_min_cls_weight = float(loss_min_cls_weight)
        self.loss_min_aff_weight = float(loss_min_aff_weight)
        self.loss_entropy_weight = float(loss_entropy_weight)

    @staticmethod
    def parse_model_output(output):
        if not isinstance(output, (tuple, list)):
            return output, None, {}
        logits = output[0]
        aff_pred = output[1] if len(output) > 1 else None
        aux = output[2] if len(output) > 2 and isinstance(output[2], dict) else {}
        return logits, aff_pred, aux

    def get_aff_loss_scale(self, epoch):
        ramp_epochs = int(self.aff_loss_ramp_epochs)
        if ramp_epochs <= 0:
            return 1.0
        return min(1.0, float(epoch + 1) / ramp_epochs)

    def forward(
        self,
        output,
        label,
        feature,
        epoch,
        training=False,
        shuffle_affective_target=False,
    ):
        logits, aff_pred, aux = self.parse_model_output(output)
        feature = feature.view(feature.size(0), -1)
        if shuffle_affective_target and training and feature.size(0) > 1:
            feature = feature[torch.randperm(feature.size(0), device=feature.device)]

        cls_vec = F.cross_entropy(logits, label, reduction="none")
        if aff_pred is None:
            aff_vec = torch.zeros_like(cls_vec)
        else:
            aff_pred = aff_pred.view(aff_pred.size(0), -1)
            if aff_pred.shape != feature.shape:
                raise ValueError(
                    "Affective output shape {} does not match target shape {}.".format(
                        tuple(aff_pred.shape),
                        tuple(feature.shape),
                    )
                )
            if self.aff_loss_type == "SmoothL1":
                aff_vec = F.smooth_l1_loss(aff_pred, feature, reduction="none").mean(dim=1)
            else:
                aff_vec = F.mse_loss(aff_pred, feature, reduction="none").mean(dim=1)

        weights = aux.get("loss_weights", None)
        if weights is None:
            weights = logits.new_full((logits.size(0), 2), 0.5)
        else:
            if weights.dim() != 2 or weights.size(0) != logits.size(0) or weights.size(1) != 2:
                raise ValueError("loss_weights must have shape [N, 2], got {}.".format(tuple(weights.shape)))
            weights = weights.to(dtype=logits.dtype, device=logits.device)
            weights = weights.clamp_min(1e-6)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        min_cls = float(self.loss_min_cls_weight)
        min_aff = float(self.loss_min_aff_weight)
        if min_cls < 0 or min_aff < 0 or min_cls + min_aff >= 1:
            raise ValueError("loss_min_cls_weight + loss_min_aff_weight must be in [0, 1).")
        min_weights = logits.new_tensor([min_cls, min_aff]).view(1, 2)
        weights = min_weights + (1.0 - min_cls - min_aff) * weights

        aff_scale = self.get_aff_loss_scale(epoch)
        loss_vec = weights[:, 0] * cls_vec + weights[:, 1] * (aff_scale * aff_vec)
        loss = loss_vec.mean()
        if self.loss_entropy_weight > 0:
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
            entropy_penalty = math.log(2.0) - entropy
            loss = loss + self.loss_entropy_weight * entropy_penalty

        metrics = {
            "loss_cls": cls_vec.mean(),
            "loss_aff": aff_vec.mean(),
            "loss_weight_cls": weights[:, 0].mean(),
            "loss_weight_aff": weights[:, 1].mean(),
            "aff_scale": logits.new_tensor(aff_scale),
        }
        return logits, loss, metrics
