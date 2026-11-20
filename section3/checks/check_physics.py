"""Independent checks of simulate.py physics (CPU, small grids).
1. No noise: the batched synchronous engine must give exactly the same final grid and time-average as a plain
   sequential (one-cell-at-a-time) abelian sandpile - the abelian property makes the order irrelevant.
2. No noise: a run's result must not depend on which other runs share its batch.
3. Noise: per-direction delivery rate must equal alpha (counted over many topplings).
4. Heights never exceed 7 during relaxation (the 'one topple per wave' exactness argument)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulate as S

def sequential(size, bg, grains):
    f = size + 2; c = f // 2
    g = np.full((f, f), bg, np.int64); g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 0
    acc = np.zeros_like(g)
    for _ in range(grains):
        g[c, c] += 1
        stack = [(c, c)]
        while stack:
            r, q = stack.pop()
            if g[r, q] < 4 or r in (0, f - 1) or q in (0, f - 1):
                continue
            g[r, q] -= 4
            for dr, dq in ((-1, 0), (1, 0), (0, 1), (0, -1)):
                rr, qq = r + dr, q + dq
                if rr in (0, f - 1) or qq in (0, f - 1):
                    continue                       # absorbed
                g[rr, qq] += 1
                if g[rr, qq] >= 4: stack.append((rr, qq))
            if g[r, q] >= 4: stack.append((r, q))
        acc += g
    return g[1:-1, 1:-1], acc[1:-1, 1:-1] / grains

size, bg, grains = 31, 2, 1500
ref_fin, ref_z = sequential(size, bg, grains)

# 1 + 2: batch of [standard, noisy, standard]
al = np.array([[1, 1, 1, 1], [0.7, 1, 0.9, 1], [1, 1, 1, 1]], float)
p = S.NumpyPile(al, size, bg, seed=1)
maxh = 0
for i in range(grains):
    p.drop()
    while p.unstable():
        p.wave(); maxh = max(maxh, p.max_height())
    p.accumulate()
z, fin = p.results(grains)
print("1. standard (in batch) final grid == sequential:", np.array_equal(fin[0], ref_fin),
      "| time-average equal:", np.allclose(z[0], ref_z))
print("2. both standard copies identical despite a noisy batch-mate:", np.array_equal(fin[0], fin[2]) and np.allclose(z[0], z[2]))
print("4. max height seen during relaxation:", maxh, "(must be <= 7)")

# mass test: expected loss per toppling = sum(1 - alpha) over directions (interior cells)
al = np.array([[0.7, 0.95, 0.5, 1.0]])
p = S.NumpyPile(al, 61, 2, seed=5)
top = 0; loss = 0
for i in range(600):
    p.drop()
    while p.unstable():
        g0 = p.grid.copy(); t = g0 >= 4
        interior = np.zeros_like(t); interior[:, 2:-2, 2:-2] = True
        if (t & ~interior).any():
            p.wave(); continue                     # skip waves touching the edge (absorbing boundary)
        n = t.sum(); p.wave()
        top += n; loss += 4 * n - (p.grid.sum() - (g0.sum() - 4 * n))
exp = (1 - al[0]).sum()
print(f"3. grains lost per toppling: measured {loss / top:.4f}  expected {exp:.4f}  over {top} topplings")
