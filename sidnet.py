"""
SIDNet: Sandpile Identification Network (Sec. 5, Fig. 10, Table 5).

Inputs
  delta      [B, H, W]            time-averaged map Delta
  win_short  [B, 16, H, W]        short window of frame-level maps Delta_t
  win_long   [B, 64, H, W]        long window, centred on the short window
Outputs
  class_logits  [B, 16]           noise class c_hat
  s_hat         [B, 4]            suppression probabilities in [0, 1]^4  (N, S, E, W)
  alpha_hat     [B, 4]            noise strengths in [0, 0.9]^4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sid_common import ALPHA_MAX, ALPHA_MIN, N_CLASSES, N_DIRS, NOISE_NAMES, direction_bits

D_MODEL = 256          # token dimension d
N_HEADS = 8
N_BLOCKS = 4
D_FF = 512
DROPOUT = 0.1
PATCH = 10             # 100 non-overlapping 10 x 10 patches
N_PATCHES = 100
N_PATCH_STATS = 14
STRIP = 15             # boundary strip width w
N_EXPLICIT = 32        # A_d (4) + 28 statistics
D_TEMP = 32            # temporal summary z_temp
D_NODE = 64            # GNN node state
GNN_ROUNDS = 3
D_EMB = 16             # embedding of the nearest noise class

PATTERNS = torch.tensor([direction_bits(n) for n in NOISE_NAMES], dtype=torch.float32)  # [16, 4]


def to_alpha(x):
    return ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * torch.sigmoid(x)


# --------------------------------------------------------------------------- explicit detector
class ExplicitDirectionDetector(nn.Module):
    """Boundary-strip scores A_d (Eq. boundary_strip) and 28 related statistics
    of the time-averaged map, followed by two two-layer MLPs (32 -> 16 -> 4)."""

    def __init__(self, strip=STRIP):
        super().__init__()
        self.w = strip
        self.dir_mlp = nn.Sequential(nn.Linear(N_EXPLICIT, 16), nn.GELU(), nn.Dropout(DROPOUT),
                                     nn.Linear(16, N_DIRS))
        self.alpha_mlp = nn.Sequential(nn.Linear(N_EXPLICIT, 16), nn.GELU(), nn.Dropout(DROPOUT),
                                       nn.Linear(16, N_DIRS))

    def features(self, x):
        B, H, W = x.shape
        w, eps = self.w, 1e-4
        strips = [x[:, :w, :], x[:, -w:, :], x[:, :, -w:], x[:, :, :w]]     # S_N, S_S, S_E, S_W
        A = torch.stack([s.mean((1, 2)) for s in strips], -1)               # A_d       [B, 4]
        sd = torch.stack([s.flatten(1).std(1) for s in strips], -1)         # strip std [B, 4]
        centre = x[:, w:-w, w:-w].mean((1, 2))
        grad = A - centre[:, None]                                          # strip-to-interior gradients
        q = torch.stack([x[:, :H // 2, :W // 2].mean((1, 2)), x[:, :H // 2, W // 2:].mean((1, 2)),
                         x[:, H // 2:, :W // 2].mean((1, 2)), x[:, H // 2:, W // 2:].mean((1, 2))],
                        -1)                                                 # quadrants NW, NE, SW, SE
        flat = x.flatten(1)
        glob = torch.stack([flat.mean(1), flat.std(1), flat.amin(1), flat.amax(1)], -1)
        aA = A.abs()
        cross = torch.stack([A[:, 0] - A[:, 1], A[:, 2] - A[:, 3],          # cross-axis differences
                             q[:, 0] - q[:, 3], q[:, 1] - q[:, 2]], -1)
        logr = torch.stack([torch.log((aA[:, 0] + eps) / (aA[:, 1] + eps)),  # log-ratios of opposite
                            torch.log((aA[:, 2] + eps) / (aA[:, 3] + eps)),  # boundary magnitudes
                            torch.log((aA[:, 0] + aA[:, 1] + eps) / (aA[:, 2] + aA[:, 3] + eps))],
                           -1).clamp(-3, 3)
        curv = torch.stack([grad[:, 0] + grad[:, 1], grad[:, 2] + grad[:, 3]], -1)  # second-order curvature
        misc = torch.stack([(q[:, 0] + q[:, 3]) - (q[:, 1] + q[:, 2]),
                            flat.abs().mean(1),
                            aA.amax(1) / (aA.mean(1) + eps)], -1)
        f = torch.cat([A, grad, sd, q, glob, cross, logr, curv, misc], -1)  # 4 + 28 = 32
        return f

    def forward(self, delta):
        f = self.features(delta)
        return torch.sigmoid(self.dir_mlp(f)), to_alpha(self.alpha_mlp(f)), f


# Indices of the direction-specific values in f, used to initialise the GNN nodes
F_A, F_GRAD, F_STD = 0, 4, 8


# --------------------------------------------------------------------------- spatial encoder
class SpatialEncoder(nn.Module):
    """Four 3 x 3 conv layers (1 -> 32 -> 64 -> 128 -> 256), each followed by batch
    normalisation, ReLU and 2 x 2 max-pooling; adaptive average pooling gives z_spa."""

    def __init__(self):
        super().__init__()
        ch, layers = [1, 32, 64, 128, 256], []
        for i in range(4):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3, padding=1), nn.BatchNorm2d(ch[i + 1]),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, delta):
        return self.net(delta.unsqueeze(1))                                 # z_spa [B, 256]


# --------------------------------------------------------------------------- patch embedding
class PatchEmbedding(nn.Module):
    """For each of the 100 patches, 14 statistics of its record across the window,
    projected linearly to d = 256."""

    def __init__(self, spike_threshold=0.1):
        super().__init__()
        self.thr = spike_threshold
        self.proj = nn.Linear(N_PATCH_STATS, D_MODEL)

    def forward(self, x):                                                   # x [B, T, H, W]
        B, T, H, W = x.shape
        n = H // PATCH
        p = x.reshape(B, T, n, PATCH, n, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(B, n * n, T, PATCH * PATCH)
        m = p.mean(-1)                                                      # patch mean per frame [B, P, T]
        h = max(T // 2, 1)
        top, bot = p[..., :PATCH * PATCH // 2], p[..., PATCH * PATCH // 2:]
        spikes = (p.abs() > self.thr).float()                               # avalanche activity
        stats = torch.stack([
            m.mean(-1), m.var(-1),                                          # temporal mean and variance
            m[..., :h].mean(-1) - m[..., h:].mean(-1),                      # half-window differences
            m[..., :h].var(-1), m[..., h:].var(-1),
            m.amax(-1), m.amin(-1),                                         # extrema
            (top.mean(-1) - bot.mean(-1)).mean(-1),                         # intra-patch spatial gradients
            (p[..., 0::2].mean(-1) - p[..., 1::2].mean(-1)).mean(-1),
            top.mean((-1, -2)), bot.mean((-1, -2)),
            p.abs().mean((-1, -2)),
            spikes.mean((-1, -2)),                                          # how often fronts pass
            (p.abs() * spikes).sum((-1, -2)) / (spikes.sum((-1, -2)) + 1e-6),  # how intensely
        ], -1)                                                              # [B, 100, 14]
        return self.proj(stats)


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(D_MODEL, N_HEADS, dropout=DROPOUT, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)
        self.ff = nn.Sequential(nn.Linear(D_MODEL, D_FF), nn.GELU(), nn.Dropout(DROPOUT),
                                nn.Linear(D_FF, D_MODEL), nn.Dropout(DROPOUT))

    def forward(self, x):
        x = self.n1(x + self.attn(x, x, x, need_weights=False)[0])
        return self.n2(x + self.ff(x))


class WindowViT(nn.Module):
    """Patch embedding + learnable positional embedding + 4 Transformer blocks."""

    def __init__(self):
        super().__init__()
        self.embed = PatchEmbedding()
        self.pos = nn.Parameter(torch.randn(1, N_PATCHES, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_BLOCKS)])
        self.norm = nn.LayerNorm(D_MODEL)

    def forward(self, win):
        t = self.embed(win) + self.pos
        for b in self.blocks:
            t = b(t)
        t = self.norm(t)
        return t, t.mean(1)                                                 # tokens [B,100,256], z [B,256]


# --------------------------------------------------------------------------- direction GNN
class DirectionGNN(nn.Module):
    """Four direction nodes in a fully connected graph, 3 rounds of message passing.
    Each node starts from (i) the patch tokens of both branches averaged over the two
    patch rows/columns nearest its boundary, (ii) its direction-specific explicit
    features and (iii) the temporal summary z_temp."""

    def __init__(self):
        super().__init__()
        d_in = 2 * D_MODEL + 3 + D_TEMP
        self.init = nn.ModuleList([nn.Linear(d_in, D_NODE) for _ in range(N_DIRS)])
        self.msg = nn.ModuleList([nn.Sequential(nn.Linear(2 * D_NODE, 2 * D_NODE), nn.GELU(),
                                                nn.Linear(2 * D_NODE, D_NODE))
                                  for _ in range(GNN_ROUNDS)])
        self.norm = nn.ModuleList([nn.LayerNorm(D_NODE) for _ in range(GNN_ROUNDS)])
        self.supp_head = nn.Linear(D_NODE, 1)
        self.alpha_head = nn.Sequential(nn.Linear(D_NODE, 32), nn.GELU(), nn.Linear(32, 1))

    @staticmethod
    def boundary_pool(tokens, d, rows=2):
        g = tokens.reshape(tokens.shape[0], 10, 10, -1)
        part = (g[:, :rows], g[:, -rows:], g[:, :, -rows:], g[:, :, :rows])[d]   # N, S, E, W
        return part.flatten(1, 2).mean(1)

    def forward(self, tok_s, tok_l, f, z_temp):
        nodes = []
        for d in range(N_DIRS):
            dir_feats = torch.stack([f[:, F_A + d], f[:, F_GRAD + d], f[:, F_STD + d]], -1)
            x = torch.cat([self.boundary_pool(tok_s, d), self.boundary_pool(tok_l, d),
                           dir_feats, z_temp], -1)
            nodes.append(self.init[d](x))
        h = torch.stack(nodes, 1)                                           # [B, 4, 64]
        for r in range(GNN_ROUNDS):
            others = (h.sum(1, keepdim=True) - h) / (N_DIRS - 1)            # messages from the other 3 nodes
            h = self.norm[r](h + self.msg[r](torch.cat([h, others], -1)))
        s_gnn = torch.sigmoid(self.supp_head(h).squeeze(-1))
        a_gnn = to_alpha(self.alpha_head(h).squeeze(-1))
        return h, s_gnn, a_gnn


# --------------------------------------------------------------------------- fusion and head
class FusionGate(nn.Module):
    """g in [0,1]^4 learned from the explicit features f and z_spa (Eq. fusion_gate)."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(N_EXPLICIT + D_MODEL, 64), nn.GELU(),
                                  nn.Linear(64, N_DIRS), nn.Sigmoid())

    def forward(self, s_exp, s_gnn, f, z_spa):
        g = self.gate(torch.cat([f, z_spa], -1))
        return g * s_exp + (1 - g) * s_gnn


class ClassificationHead(nn.Module):
    """Concatenation of s_hat, the embedding of the nearest noise class, z_s, z_l, z_spa,
    the GNN node states and f (D_cat = 4 + 16 + 3*256 + 4*64 + 32 = 1,076), a three-layer
    MLP (1076 -> 256 -> 128 -> 16) and a linear residual shortcut from s_hat."""

    D_CAT = N_DIRS + D_EMB + 3 * D_MODEL + N_DIRS * D_NODE + N_EXPLICIT

    def __init__(self):
        super().__init__()
        self.register_buffer("patterns", PATTERNS.clone())
        self.class_emb = nn.Embedding(N_CLASSES, D_EMB)
        self.mlp = nn.Sequential(nn.Linear(self.D_CAT, 256), nn.GELU(), nn.Dropout(DROPOUT),
                                 nn.Linear(256, 128), nn.GELU(), nn.Dropout(DROPOUT),
                                 nn.Linear(128, N_CLASSES))
        self.shortcut = nn.Linear(N_DIRS, N_CLASSES)

    def nearest_class(self, s_hat):
        bits = (s_hat > 0.5).float()                                        # threshold at 0.5
        return (bits[:, None, :] == self.patterns[None]).all(-1).float().argmax(1)

    def forward(self, s_hat, z_s, z_l, z_spa, nodes, f):
        e = self.class_emb(self.nearest_class(s_hat))
        x = torch.cat([s_hat, e, z_s, z_l, z_spa, nodes.flatten(1), f], -1)
        return self.mlp(x) + self.shortcut(s_hat)


class SIDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.explicit = ExplicitDirectionDetector()
        self.spatial = SpatialEncoder()
        self.vit_short = WindowViT()
        self.vit_long = WindowViT()
        self.temporal = nn.Linear(2 * D_MODEL, D_TEMP)                      # z_temp from [z_s, z_l]
        self.gnn = DirectionGNN()
        self.fusion = FusionGate()
        self.head = ClassificationHead()
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, delta, win_short, win_long):
        s_exp, a_exp, f = self.explicit(delta)
        z_spa = self.spatial(delta)
        tok_s, z_s = self.vit_short(win_short)
        tok_l, z_l = self.vit_long(win_long)
        z_temp = self.temporal(torch.cat([z_s, z_l], -1))
        nodes, s_gnn, a_gnn = self.gnn(tok_s, tok_l, f, z_temp)
        s_hat = self.fusion(s_exp, s_gnn, f, z_spa)
        alpha_hat = 0.4 * a_exp + 0.6 * a_gnn                               # fixed weights
        logits = self.head(s_hat, z_s, z_l, z_spa, nodes, f)
        return logits, s_hat, alpha_hat


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = SIDNet()
    print(f"SIDNet: {count_parameters(m):,} trainable parameters, D_cat = {ClassificationHead.D_CAT}")
    B = 2
    out = m(torch.randn(B, 100, 100), torch.randn(B, 16, 100, 100), torch.randn(B, 64, 100, 100))
    print("class_logits", tuple(out[0].shape), " s_hat", tuple(out[1].shape),
          " alpha_hat", tuple(out[2].shape))
