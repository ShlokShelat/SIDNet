"""The same checks on the GPU engine (TorchPile): no-noise result == plain sequential sandpile; batch-mates don't
interfere; noise loses grains at the stated rate."""
import sys, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulate as S
from check_physics import sequential

dev = torch.device("cuda")
size, bg, grains = 31, 2, 1500
ref_fin, ref_z = sequential(size, bg, grains)
al = np.array([[1, 1, 1, 1], [0.7, 1, 0.9, 1], [1, 1, 1, 1]], float)
p = S.TorchPile(al, size, bg, 1, dev)
for i in range(grains):
    p.drop(); S.relax(p); p.accumulate()
z, fin = p.results(grains)
print("GPU 1. standard == sequential:", np.array_equal(fin[0], ref_fin), np.allclose(z[0], ref_z))
print("GPU 2. batch-mates don't interfere:", np.array_equal(fin[0], fin[2]) and np.allclose(z[0], z[2]))
al = np.array([[0.7, 0.95, 0.5, 1.0]])
p = S.TorchPile(al, 61, 2, 5, dev)
top = loss = 0
for i in range(600):
    p.drop()
    while p.unstable():
        g0 = p.grid.clone(); t = g0 >= 4
        if t[:, :2].any() or t[:, -2:].any() or t[:, :, :2].any() or t[:, :, -2:].any():
            p.wave(); continue
        n = int(t.sum()); p.wave()
        top += n; loss += 4 * n - int(p.grid.sum() - (g0.sum() - 4 * n))
print(f"GPU 3. grains lost per toppling: {loss / top:.4f} expected {(1 - al[0]).sum():.4f} over {top}")
