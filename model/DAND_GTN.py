import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def linear_init(fc, std=0.02):
    nn.init.normal_(fc.weight, 0, std)
    if fc.bias is not None:
        nn.init.constant_(fc.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def init_module(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            conv_init(m)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bn_init(m, 1)
        elif isinstance(m, nn.Linear):
            linear_init(m)


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node), dtype=np.float32)
    for i, j in link:
        A[j, i] = 1.0
    return A


def normalize_digraph(A):
    degree = np.sum(A, axis=0)
    Dn = np.zeros((A.shape[1], A.shape[1]), dtype=np.float32)
    for i, value in enumerate(degree):
        if value > 0:
            Dn[i, i] = value ** -1
    return np.dot(A, Dn).astype(np.float32)


def build_emotion_gait_pose_A(num_point=16):
    if num_point != 16:
        raise ValueError("Emotion-Gait data-aware adjacency expects 16 joints")

    self_link = [(i, i) for i in range(num_point)]
    inward = [
        (0, 1),
        (1, 2),
        (2, 3),
        (2, 4),
        (4, 5),
        (5, 6),
        (2, 7),
        (7, 8),
        (8, 9),
        (0, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
    ]
    outward = [(j, i) for i, j in inward]
    return np.stack(
        (
            edge2mat(self_link, num_point),
            normalize_digraph(edge2mat(inward, num_point)),
            normalize_digraph(edge2mat(outward, num_point)),
        )
    )


def resolve_num_heads(d_model):
    for n_head in (8, 4, 2, 1):
        if d_model % n_head == 0:
            return n_head
    return 1


def _as_stage_list(value, num_stages, default):
    if value is None:
        return [default] * num_stages
    if isinstance(value, int):
        return [value] * num_stages
    values = list(value)
    if not values:
        return [default] * num_stages
    if len(values) < num_stages:
        values = values + [values[-1]] * (num_stages - len(values))
    return values[:num_stages]


class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.num_subset = A.shape[0]
        self.A = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(self.num_subset))
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, 1, bias=False) for _ in range(self.num_subset)]
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        init_module(self)

    def forward(self, x):
        y = None
        for subset_idx in range(self.num_subset):
            adj = self.A[subset_idx]
            identity = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
            mixed_adj = adj + self.alpha[subset_idx] * identity
            z = torch.einsum("nctu,vu->nctv", x, mixed_adj)
            z = self.convs[subset_idx](z)
            y = z if y is None else y + z
        return self.relu(self.bn(y))


class PartAwareGate(nn.Module):
    def __init__(self, channels, num_joints=16):
        super().__init__()
        self.part_groups = self.get_part_groups(num_joints)
        hidden = max(channels // 4, 16)
        self.part_mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(True),
            nn.Conv1d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        init_module(self)

    @staticmethod
    def get_part_groups(num_joints):
        if num_joints == 16:
            return [
                [0, 1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15],
            ]
        chunk = max(1, num_joints // 5)
        groups = []
        for start in range(0, num_joints, chunk):
            groups.append(list(range(start, min(start + chunk, num_joints))))
        return groups

    def forward(self, x):
        n, c, _, v = x.size()
        part_descriptors = []
        for group in self.part_groups:
            part_descriptors.append(x[:, :, :, group].mean(dim=-1).mean(dim=-1, keepdim=True))
        part_descriptors = torch.cat(part_descriptors, dim=-1)
        part_weights = self.part_mlp(part_descriptors)

        gate = x.new_zeros(n, c, v)
        for part_idx, group in enumerate(self.part_groups):
            weight = part_weights[:, :, part_idx].unsqueeze(-1)
            gate[:, :, group] = weight.expand(-1, -1, len(group))
        return x * (1 + gate.unsqueeze(2))


class SGCBranch(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, num_joints=16):
        super().__init__()
        self.gcn = AdaptiveGraphConv(in_channels, out_channels, A)
        self.tcn = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(3, 1),
                stride=(stride, 1),
                padding=(1, 0),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.part_gate = PartAwareGate(out_channels, num_joints=num_joints)
        self.relu = nn.ReLU(True)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
            init_module(self.residual)

        init_module(self.tcn)

    def forward(self, x):
        y = self.gcn(x)
        y = self.tcn(y)
        y = self.part_gate(y)
        return self.relu(y + self.residual(x))


class TABranch(nn.Module):
    def __init__(self, channels, stride=1, drop_prob=0.1):
        super().__init__()
        n_head = resolve_num_heads(channels)
        self.attn = nn.MultiheadAttention(channels, n_head, dropout=drop_prob, batch_first=True)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.drop = nn.Dropout(drop_prob)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(channels * 2, channels),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, stride=(stride, 1), bias=False),
            nn.BatchNorm2d(channels),
        )
        init_module(self.ffn)
        init_module(self.proj)

    def forward(self, x):
        n, c, t, v = x.size()
        seq = x.permute(0, 3, 2, 1).contiguous().view(n * v, t, c)
        attn_out, _ = self.attn(seq, seq, seq, need_weights=False)
        seq = self.norm1(seq + self.drop(attn_out))
        seq = self.norm2(seq + self.drop(self.ffn(seq)))
        seq = seq.view(n, v, t, c).permute(0, 3, 2, 1).contiguous()
        return self.proj(seq)


class DynamicNodeConcentration(nn.Module):
    def __init__(self, channels, num_joints, num_relay, attn_ratio=0.5):
        super().__init__()
        hidden = max(int(channels * attn_ratio), 16)
        self.num_joints = num_joints
        self.num_relay = max(1, min(int(num_relay), num_joints))
        self.scale = hidden ** -0.5

        self.spatial_q = nn.Linear(channels, hidden, bias=False)
        self.spatial_k = nn.Linear(channels, hidden, bias=False)
        self.phi = nn.Linear(channels, hidden, bias=False)
        self.psi = nn.Linear(channels, hidden, bias=False)
        self.act = nn.GELU()
        init_module(self)

    def forward(self, x):
        _, _, v, c = x.size()
        k_relay = min(self.num_relay, v)

        q = self.spatial_q(x)
        k = self.spatial_k(x)
        spatial_score = torch.einsum("btid,btjd->btij", q, k) * self.scale
        spatial_attn = torch.softmax(spatial_score, dim=-1)

        importance = spatial_attn.mean(dim=1).mean(dim=1)
        anchor_index = torch.topk(importance, k=k_relay, dim=-1).indices

        x_summary = x.mean(dim=1)
        gather_index = anchor_index.unsqueeze(-1).expand(-1, -1, c)
        anchor = torch.gather(x_summary, dim=1, index=gather_index)

        node_emb = self.act(self.phi(x_summary))
        anchor_emb = self.act(self.psi(anchor))
        group_score = torch.einsum("bvd,bkd->bvk", node_emb, anchor_emb) * self.scale
        soft_group = torch.softmax(group_score, dim=-1)
        hard_group = F.one_hot(soft_group.argmax(dim=-1), num_classes=k_relay).type_as(soft_group)
        group = hard_group.detach() - soft_group.detach() + soft_group

        counts = group.sum(dim=1).clamp_min(1.0)
        relay = torch.einsum("bvk,btvc->bktc", group, x)
        relay = relay / counts[:, :, None, None]
        return relay, group, counts, spatial_attn


class TemporalNodeDiffusion(nn.Module):
    def __init__(self, channels, attn_ratio=0.5, drop_prob=0.1):
        super().__init__()
        hidden = max(int(channels * attn_ratio), 16)
        self.scale = hidden ** -0.5

        self.q_proj = nn.Linear(channels, hidden, bias=False)
        self.k_proj = nn.Linear(channels, hidden, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        self.drop = nn.Dropout(drop_prob)
        init_module(self)

    def forward(self, x, relay, group, counts):
        value = self.v_proj(x)
        relay_value = torch.einsum("bvk,btvc->bktc", group, value)
        relay_value = relay_value / counts[:, :, None, None]

        q = self.q_proj(relay)
        k = self.k_proj(relay)
        temporal_score = torch.einsum("bktd,bksd->bkts", q, k) * self.scale
        temporal_attn = torch.softmax(temporal_score, dim=-1)
        temporal_attn = self.drop(temporal_attn)

        relay_context = torch.einsum("bkts,bksc->bktc", temporal_attn, relay_value)
        diffused = torch.einsum("bvk,bktc->btvc", group, relay_context)
        return self.out_proj(diffused), temporal_attn


class DNDBlock(nn.Module):
    def __init__(
        self,
        channels,
        num_joints=16,
        num_relay=8,
        temporal_window=8,
        use_temporal_shift=False,
        drop_prob=0.1,
        init_scale=0.1,
    ):
        super().__init__()
        self.temporal_window = temporal_window
        self.use_temporal_shift = use_temporal_shift

        self.concentration = DynamicNodeConcentration(channels, num_joints, num_relay)
        self.diffusion = TemporalNodeDiffusion(channels, drop_prob=drop_prob)
        self.diff_norm = nn.BatchNorm2d(channels)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(True),
            nn.Dropout(drop_prob),
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(True)
        self.gamma_diff = nn.Parameter(torch.tensor(float(init_scale)))
        self.gamma_ffn = nn.Parameter(torch.tensor(float(init_scale)))

        init_module(self.gate)
        init_module(self.ffn)
        bn_init(self.diff_norm, 1)

    def _core(self, x):
        x_nodes = x.permute(0, 2, 3, 1).contiguous()
        relay, group, counts, spatial_attn = self.concentration(x_nodes)
        diffused, temporal_attn = self.diffusion(x_nodes, relay, group, counts)
        diffused = diffused.permute(0, 3, 1, 2).contiguous()
        diffused = self.diff_norm(diffused)

        gate = self.gate(torch.cat([x, diffused], dim=1))
        y = self.relu(x + self.gamma_diff * gate * diffused)
        y = self.relu(y + self.gamma_ffn * self.ffn(y))
        aux = {
            "relay": relay,
            "group": group,
            "spatial_attn": spatial_attn,
            "temporal_attn": temporal_attn,
        }
        return y, aux

    def forward(self, x, return_aux=False):
        n, c, t, v = x.size()
        window = int(self.temporal_window) if self.temporal_window else 0
        if window <= 1 or window >= t:
            y, aux = self._core(x)
            return (y, aux) if return_aux else y

        shift = window // 2 if self.use_temporal_shift else 0
        if shift > 0:
            x_work = torch.roll(x, shifts=-shift, dims=2)
        else:
            x_work = x

        pad_len = (window - (t % window)) % window
        if pad_len:
            x_work = F.pad(x_work, (0, 0, 0, pad_len))

        x_work = x_work.contiguous()
        t_pad = x_work.size(2)
        num_windows = t_pad // window
        x_win = x_work.view(n, c, num_windows, window, v)
        x_win = x_win.permute(0, 2, 1, 3, 4).contiguous().view(n * num_windows, c, window, v)
        y_win, aux = self._core(x_win)

        y = y_win.view(n, num_windows, c, window, v)
        y = y.permute(0, 2, 1, 3, 4).contiguous().view(n, c, t_pad, v)
        if pad_len:
            y = y[:, :, :t, :]
        if shift > 0:
            y = torch.roll(y, shifts=shift, dims=2)

        return (y, aux) if return_aux else y


class DSAW(nn.Module):
    def __init__(self, channels, hidden_ratio=0.25, drop_prob=0.0, preserve_magnitude=True):
        super().__init__()
        hidden = max(int(channels * hidden_ratio), 16)
        self.preserve_magnitude = preserve_magnitude
        self.score = nn.Sequential(
            nn.Linear(channels * 4, hidden, bias=False),
            nn.ReLU(True),
            nn.Dropout(drop_prob),
            nn.Linear(hidden, 2, bias=True),
        )
        init_module(self.score)
        nn.init.constant_(self.score[-1].weight, 0)
        nn.init.constant_(self.score[-1].bias, 0)

    def forward(self, pose_feat, motion_feat):
        joint_context = torch.cat(
            [
                pose_feat,
                motion_feat,
                torch.abs(pose_feat - motion_feat),
                pose_feat * motion_feat,
            ],
            dim=1,
        )
        weights = torch.softmax(self.score(joint_context), dim=1)
        scale = weights.size(1) if self.preserve_magnitude else 1.0
        pose_weighted = pose_feat * weights[:, 0:1] * scale
        motion_weighted = motion_feat * weights[:, 1:2] * scale
        return pose_weighted, motion_weighted, weights


class DLAHead(nn.Module):
    """Dynamic loss-aware head: DSAW feature fusion plus ALF loss utilities."""

    def __init__(
        self,
        final_channels,
        num_class=4,
        aff_dim=1488,
        drop_out=0.0,
        adaptive_weight_hidden_ratio=0.25,
        adaptive_weight_drop=0.0,
        head_hidden_ratio=0.75,
        loss_gate_hidden_ratio=0.5,
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

        self.stream_weight = DSAW(
            final_channels,
            hidden_ratio=adaptive_weight_hidden_ratio,
            drop_prob=adaptive_weight_drop,
        )

        context_dim = final_channels * 3
        head_hidden = max(int(context_dim * head_hidden_ratio), final_channels)
        gate_hidden = max(int(context_dim * loss_gate_hidden_ratio), 64)

        self.out_drop = nn.Dropout(drop_out) if drop_out else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim)
        self.classifier = nn.Linear(context_dim, num_class)
        self.aff_head = nn.Sequential(
            nn.Linear(context_dim, head_hidden),
            nn.ReLU(True),
            nn.Dropout(drop_out),
            nn.Linear(head_hidden, aff_dim),
        )
        self.loss_gate = nn.Sequential(
            nn.Linear(context_dim, gate_hidden),
            nn.ReLU(True),
            nn.Dropout(drop_out),
            nn.Linear(gate_hidden, 2),
        )

        linear_init(self.classifier, math.sqrt(2.0 / context_dim))
        init_module(self.aff_head)
        init_module(self.loss_gate)
        nn.init.constant_(self.loss_gate[-1].weight, 0)
        nn.init.constant_(self.loss_gate[-1].bias, 0)

    def forward(self, pose_repr, motion_repr, fused_repr):
        pose_weighted, motion_weighted, stream_weights = self.stream_weight(
            pose_repr,
            motion_repr,
        )
        final_context = torch.cat([pose_weighted, motion_weighted, fused_repr], dim=1)
        final_context = self.context_norm(final_context)
        final_context = self.out_drop(final_context)

        logits = self.classifier(final_context)
        aff_pred = self.aff_head(final_context)
        loss_weights = torch.softmax(self.loss_gate(final_context), dim=1)

        aux = {
            "stream_weights": stream_weights,
            "loss_weights": loss_weights,
            "final_context": final_context,
            "pose_weighted": pose_weighted,
            "motion_weighted": motion_weighted,
        }
        return logits, aff_pred, aux

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

    def compute_loss(
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


class CrossStreamEncoder(nn.Module):
    def __init__(self, d_model, n_head=8, ffn_dim=None, dropout=0.1):
        super().__init__()
        ffn_dim = ffn_dim or 2 * d_model
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        init_module(self.ffn)

    def forward(self, x, y):
        n, c, t, v = x.size()
        x_seq = x.permute(0, 3, 2, 1).contiguous().view(n * v, t, c)
        y_seq = y.permute(0, 3, 2, 1).contiguous().view(n * v, t, c)

        dx, _ = self.attn(x_seq, y_seq, y_seq, need_weights=False)
        dy, _ = self.attn(y_seq, x_seq, x_seq, need_weights=False)
        x_seq = self.norm1(x_seq + self.drop1(dx))
        y_seq = self.norm1(y_seq + self.drop1(dy))
        x_seq = self.norm2(x_seq + self.drop2(self.ffn(x_seq)))
        y_seq = self.norm2(y_seq + self.drop2(self.ffn(y_seq)))

        x_out = x_seq.view(n, v, t, c).permute(0, 3, 2, 1).contiguous()
        y_out = y_seq.view(n, v, t, c).permute(0, 3, 2, 1).contiguous()
        return x_out, y_out


class CSAFBlock(nn.Module):
    def __init__(self, d_model, num_joints=16, drop_prob=0.1, init_scale=0.1):
        super().__init__()
        self.transformer = CrossStreamEncoder(
            d_model=d_model,
            n_head=resolve_num_heads(d_model),
            ffn_dim=2 * d_model,
            dropout=drop_prob,
        )
        self.spatial_att_p = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_joints, num_joints, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(num_joints, num_joints, 1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_att_m = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_joints, num_joints, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(num_joints, num_joints, 1, bias=False),
            nn.Sigmoid(),
        )
        self.gamma_cross = nn.Parameter(torch.tensor(float(init_scale)))
        self.gamma_spatial = nn.Parameter(torch.tensor(float(init_scale)))
        init_module(self.spatial_att_p)
        init_module(self.spatial_att_m)

    def forward(self, x, y):
        dx, dy = self.transformer(x, y)
        x, y = x + self.gamma_cross * dx, y + self.gamma_cross * dy
        att_p = self.spatial_att_p(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        att_m = self.spatial_att_m(y.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        x, y = x + self.gamma_spatial * y * att_m, y + self.gamma_spatial * x * att_p
        return x, y


class STNDBlock(nn.Module):
    """STND-Block: spatial graph branch + temporal attention branch + dynamic node diffusion."""

    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        stride=1,
        residual=True,
        num_joints=16,
        num_relay=8,
        temporal_window=12,
        use_temporal_shift=False,
        drop_prob=0.1,
        init_scale=0.1,
    ):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.spatial = SGCBranch(
            out_channels,
            out_channels,
            A,
            stride=stride,
            residual=False,
            num_joints=num_joints,
        )
        self.temporal = TABranch(
            out_channels,
            stride=stride,
            drop_prob=drop_prob,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.dndt = DNDBlock(
            out_channels,
            num_joints=num_joints,
            num_relay=num_relay,
            temporal_window=temporal_window,
            use_temporal_shift=use_temporal_shift,
            drop_prob=drop_prob,
            init_scale=init_scale,
        )
        self.relu = nn.ReLU(True)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
            init_module(self.residual)

        init_module(self.pre)
        init_module(self.fuse)

    def forward(self, x):
        x_proj = self.pre(x)
        spatial_feat = self.spatial(x_proj)
        temporal_feat = self.temporal(x_proj)
        y = self.fuse(torch.cat([spatial_feat, temporal_feat], dim=1))
        y = self.dndt(y)
        return self.relu(y + self.residual(x))


class Model(nn.Module):
    def __init__(
        self,
        num_class=4,
        num_point=16,
        graph=None,
        graph_args=dict(),
        in_channels_p=3,
        in_channels_m=8,
        stage_dims=(64, 64, 128, 128, 256, 256),
        stage_strides=(1, 1, 2, 1, 2, 1),
        fusion_after=(2, 4, 6),
        aff_dim=1488,
        drop_out=0.0,
        dndt_relay_nodes=(8, 8, 4, 4, 2, 2),
        dndt_temporal_window=12,
        dndt_drop=0.1,
        dndt_init_scale=0.1,
        use_data_adjacency=True,
        use_csta_fusion=True,
        adaptive_weight_hidden_ratio=0.25,
        adaptive_weight_drop=0.0,
        head_hidden_ratio=0.75,
        loss_gate_hidden_ratio=0.5,
        aff_loss_type="MSE",
        aff_loss_ramp_epochs=10,
        loss_min_cls_weight=0.6,
        loss_min_aff_weight=0.05,
        loss_entropy_weight=0.01,
        **kwargs
    ):
        super().__init__()
        if "use_msh_fusion" in kwargs:
            use_csta_fusion = kwargs.pop("use_msh_fusion")
        self.num_class = num_class
        self.num_point = num_point
        self.aff_dim = aff_dim
        self.stage_dims = list(stage_dims)
        self.stage_strides = list(stage_strides)
        self.fusion_after = [int(pos) for pos in fusion_after]
        self.use_csta_fusion = bool(use_csta_fusion)

        if len(self.stage_dims) != 6:
            raise ValueError("DAND-GTN expects six stages.")
        if len(self.stage_strides) != len(self.stage_dims):
            raise ValueError("stage_strides must have the same length as stage_dims.")
        if len(self.fusion_after) != 3 or self.fusion_after != [2, 4, 6]:
            raise ValueError("DAND-GTN uses exactly three CSAF-Block fusions after layers 2, 4 and 6.")
        if self.use_csta_fusion and num_point != 16:
            raise ValueError("CSAF-Block expects 16 joints.")
        if graph is None and not use_data_adjacency:
            raise ValueError("graph must be provided when use_data_adjacency is False.")

        if use_data_adjacency:
            self.graph = None
            A = build_emotion_gait_pose_A(num_point)
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
            graph_A, _ = self.graph.A
            A = graph_A.mean(axis=0)

        self.data_bn_p = nn.BatchNorm1d(in_channels_p * num_point)
        self.data_bn_m = nn.BatchNorm1d(in_channels_m * num_point)

        self.pose_stages = nn.ModuleList()
        self.motion_stages = nn.ModuleList()
        relay_nodes = _as_stage_list(dndt_relay_nodes, len(self.stage_dims), 2)

        prev_pose_channels = in_channels_p
        prev_motion_channels = in_channels_m
        for stage_idx, (out_channels, stride) in enumerate(zip(self.stage_dims, self.stage_strides)):
            stage_kwargs = dict(
                A=A,
                stride=stride,
                residual=stage_idx != 0,
                num_joints=num_point,
                num_relay=relay_nodes[stage_idx],
                temporal_window=dndt_temporal_window,
                use_temporal_shift=(stage_idx % 2) == 1,
                drop_prob=dndt_drop,
                init_scale=dndt_init_scale,
            )
            self.pose_stages.append(
                STNDBlock(prev_pose_channels, out_channels, **stage_kwargs)
            )
            self.motion_stages.append(
                STNDBlock(prev_motion_channels, out_channels, **stage_kwargs)
            )
            prev_pose_channels = out_channels
            prev_motion_channels = out_channels

        self.fusions = nn.ModuleDict()
        if self.use_csta_fusion:
            for point in self.fusion_after:
                self.fusions[str(point)] = CSAFBlock(self.stage_dims[point - 1])

        final_channels = self.stage_dims[-1]
        self.dla_head = DLAHead(
            final_channels=final_channels,
            num_class=num_class,
            aff_dim=aff_dim,
            drop_out=drop_out,
            adaptive_weight_hidden_ratio=adaptive_weight_hidden_ratio,
            adaptive_weight_drop=adaptive_weight_drop,
            head_hidden_ratio=head_hidden_ratio,
            loss_gate_hidden_ratio=loss_gate_hidden_ratio,
            aff_loss_type=aff_loss_type,
            aff_loss_ramp_epochs=aff_loss_ramp_epochs,
            loss_min_cls_weight=loss_min_cls_weight,
            loss_min_aff_weight=loss_min_aff_weight,
            loss_entropy_weight=loss_entropy_weight,
        )

        bn_init(self.data_bn_p, 1)
        bn_init(self.data_bn_m, 1)

    def _preprocess(self, x, in_channels, bn):
        n, _, t, v, m = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(n, m * v * in_channels, t)
        x = bn(x)
        x = x.view(n, m, v, in_channels, t).permute(0, 1, 3, 4, 2).contiguous()
        return x.view(n * m, in_channels, t, v)

    @staticmethod
    def _global_pool(x, n, m):
        return x.view(n, m, x.size(1), -1).mean(3).mean(1)

    def forward(self, x_p, x_m, label=None, get_hidden_feat=False, **kwargs):
        n, c_p, _, _, m = x_p.size()
        _, c_m, _, _, _ = x_m.size()

        x_p = self._preprocess(x_p, c_p, self.data_bn_p)
        x_m = self._preprocess(x_m, c_m, self.data_bn_m)

        pose_stage_feats = []
        motion_stage_feats = []
        fusion_stage_feats = []

        for stage_idx, (pose_stage, motion_stage) in enumerate(
            zip(self.pose_stages, self.motion_stages),
            start=1,
        ):
            x_p = pose_stage(x_p)
            x_m = motion_stage(x_m)

            if self.use_csta_fusion and str(stage_idx) in self.fusions:
                x_p, x_m = self.fusions[str(stage_idx)](x_p, x_m)
                if get_hidden_feat:
                    fusion_stage_feats.append(0.5 * (x_p + x_m))

            if get_hidden_feat:
                pose_stage_feats.append(x_p)
                motion_stage_feats.append(x_m)

        fused_feat = 0.5 * (x_p + x_m)
        pose_repr = self._global_pool(x_p, n, m)
        motion_repr = self._global_pool(x_m, n, m)
        fused_repr = self._global_pool(fused_feat, n, m)

        logits, aff_pred, aux = self.dla_head(pose_repr, motion_repr, fused_repr)
        aux.update(
            {
                "pose": pose_repr,
                "motion": motion_repr,
                "fusion": fused_repr,
            }
        )
        if get_hidden_feat:
            aux.update(
                {
                    "pose_stage_feats": pose_stage_feats,
                    "motion_stage_feats": motion_stage_feats,
                    "fusion_stage_feats": fusion_stage_feats,
                }
            )

        return logits, aff_pred, aux
