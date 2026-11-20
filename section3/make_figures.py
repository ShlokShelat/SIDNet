"""
Figures of Sec. 3 from the 102 illustration runs in section3/runs/.

  Fig. 1        FIG1.png              standard sandpile, final configuration, z legend
  Figs. 3, 4    M_*.png               space-time picture panels (uniform / non-uniform regime)
  Fig. 5        S31_uniN.pdf          class N: picture, reference and step comparisons, imbalance maps
  Fig. 6        S31_uniNS_NSE.pdf     classes NS and NSE
  Fig. 7        S31_nonuni.pdf        three non-uniform configurations
  Fig. 8        S31_chart_shift.pdf   centre shift of all uniform classes
  Fig. 9        S31_chart_both.pdf    area vs total removal; shift vs net push

Fig. 2 (the five noise families) is a schematic drawing and is not generated from data.

Usage
  python section3/make_figures.py --out figures_section3
"""
import argparse, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Circle, FancyArrow
import matplotlib.patheffects as pe
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs') + os.sep
C0 = 250
DIV = LinearSegmentedColormap.from_list('strong_div', [
    '#001a66', '#0050c8', '#4da3ff', '#ffffff', '#ff8040', '#d01010', '#5c0000'])
PAL = np.array([(13, 27, 42), (0, 180, 216), (255, 183, 3), (255, 0, 110)], np.uint8)
GREEN = '#00ff88'
FLOOR = 0.6
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8,
                     'mathtext.fontset': 'dejavusans', 'savefig.dpi': 600,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
TEXTW = 7.16                      # \textwidth of cas-dc in inches

def load(k):
    d = np.load(R + k + '__s0.npz'); return d['zbar'], d['final'].astype(int)
def zbar(k): return load(k)[0]
def delta(a, b): return (a - b) / (b + 1.0)
def win(a, h): return a[C0 - h:C0 + h + 1, C0 - h:C0 + h + 1]
def up(a, px=1400):               # repeat each site so that the map is about px pixels wide
    k = max(1, int(math.ceil(px / a.shape[0])))
    return np.kron(a, np.ones((k, k) + (1,) * (a.ndim - 2))) if a.ndim == 3 else np.kron(a, np.ones((k, k)))
def mirror_ns(d):                 # Delta minus Delta at the mirror site (o -> 2*C0 - o)
    m = np.zeros_like(d); m[1:, :] = d[1:, :][::-1, :]; return d - m
def mirror_ew(d):
    m = np.zeros_like(d); m[:, 1:] = d[:, 1:][:, ::-1]; return d - m
def lim_window(a, h): return np.percentile(np.abs(win(a, h)), 99)
def lim_nonzero(a, h):
    x = np.abs(win(a, h)); return np.percentile(x[x > 0], 99)

def geometry(z):
    rr, cc = np.nonzero(np.abs(z - 2.0) > 1e-12)
    return dict(top=rr.min() - C0, bot=rr.max() - C0, lef=cc.min() - C0, rig=cc.max() - C0,
                dy=rr.mean() - C0, dx=cc.mean() - C0)

EV = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}   # (x, y) with y pointing down (south)
def alphas_of(key):                # U_<dirs>_<1-alpha>: every listed direction has the same strength
    p = key[2:].split('_')
    if key.startswith('U_'): return {d: 1 - float(p[1]) for d in p[0]}
    return {p[i]: 1 - float(p[i + 1]) for i in range(0, len(p), 2)}
def net_push(key):                 # v = -sum (1 - alpha_d) e_d, in (x, y-down) coordinates
    a = alphas_of(key)
    return (-sum((1 - v) * EV[d][0] for d, v in a.items()), -sum((1 - v) * EV[d][1] for d, v in a.items()))

def compass_label(g):
    s = math.hypot(g['dy'], g['dx'])
    if s < FLOOR: return 'shift %.1f (below %.1f)' % (s, FLOOR), None
    if abs(g['dy']) >= abs(g['dx']): name, u = ('south', (0, 1)) if g['dy'] > 0 else ('north', (0, -1))
    else: name, u = ('east', (1, 0)) if g['dx'] > 0 else ('west', (-1, 0))
    return 'shift %.1f %s' % (s, name), u

def frame(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(0.6)

def picture(ax, key, h, title, true_angle=False, push=False):
    z, fin = load(key); g = geometry(z)
    img = PAL[np.clip(win(fin, h), 0, 3)]
    ax.imshow(up(img).astype(np.uint8), interpolation='none', extent=[-h - .5, h + .5, h + .5, -h - .5])
    ax.plot([-h - .5, h + .5], [0, 0], color='white', lw=0.8, ls=(0, (3, 2)))
    ax.plot([0, 0], [-h - .5, h + .5], color='white', lw=0.8, ls=(0, (3, 2)))
    ax.add_patch(Rectangle((g['lef'] - .5, g['top'] - .5), g['rig'] - g['lef'] + 1, g['bot'] - g['top'] + 1,
                           fill=False, ec=GREEN, lw=1.3))
    label, u = compass_label(g)
    s = math.hypot(g['dy'], g['dx'])
    hw = 0.15 * h; hl = 0.13 * h; w = 0.045 * h
    if push:                       # direction of the net push: dashed green line, longer than the white arrow
        vx, vy = net_push(key); n = math.hypot(vx, vy); L = 0.80 * h
        ex, ey = L * vx / n, L * vy / n
        ax.plot([0, 0.9 * ex], [0, 0.9 * ey], color=GREEN, lw=2.0, ls=(0, (2.2, 1.3)), zorder=4,
                path_effects=[pe.withStroke(linewidth=3.2, foreground='black')])
        ax.add_patch(FancyArrow(0.86 * ex, 0.86 * ey, 0.14 * ex, 0.14 * ey, width=0, head_width=hw,
                                head_length=hl, length_includes_head=True, fc=GREEN, ec='black', lw=0.6, zorder=4))
    if u is not None:              # direction of the centre shift (fixed length; the dot marks the centre)
        if true_angle: ux, uy = g['dx'] / s, g['dy'] / s
        else: ux, uy = u
        L = 0.52 * h
        ax.add_patch(FancyArrow(0, 0, ux * L, uy * L, width=w, head_width=hw, head_length=hl,
                                length_includes_head=True, fc='white', ec='black', lw=0.6, zorder=5))
    ax.add_patch(Circle((g['dx'], g['dy']), 0.06 * h, fc=GREEN, ec='black', lw=0.6, zorder=6))
    ax.text(0, h - 0.05 * h, label, ha='center', va='bottom', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.2', fc='#fff8e6', ec='black', lw=0.5), zorder=7)
    ax.set_xlim(-h - .5, h + .5); ax.set_ylim(h + .5, -h - .5)
    frame(ax); ax.set_title(title, pad=3); ax.set_xlabel('zoom $\\pm$%d sites' % h, labelpad=2)
    return g

def dmap(fig, ax, cax, d, h, vmax, title):
    im = ax.imshow(up(win(d, h)), cmap=DIV, vmin=-vmax, vmax=vmax, interpolation='none',
                   extent=[-h - .5, h + .5, h + .5, -h - .5])
    frame(ax); ax.set_title(title, pad=3); ax.set_xlabel('zoom $\\pm$%d sites' % h, labelpad=2)
    cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    cb.outline.set_linewidth(0.5)

def grid(nrows, ncols, s, top=0.22, between=0.50, bottom=0.27, cbw=0.07, gap=0.05, cbl=0.30):
    H = top + nrows * s + (nrows - 1) * between + bottom
    fig = plt.figure(figsize=(TEXTW, H)); cw = TEXTW / ncols; axes = []
    for i in range(nrows):
        y0 = H - top - (i + 1) * s - i * between; row = []
        for j in range(ncols):
            x0 = j * cw + (cw - s - gap - cbw - cbl) / 2
            ax = fig.add_axes([x0 / TEXTW, y0 / H, s / TEXTW, s / H])
            cax = fig.add_axes([(x0 + s + gap) / TEXTW, y0 / H, cbw / TEXTW, s / H])
            row.append((ax, cax))
        axes.append(row)
    return fig, axes

# ---------------- Fig. 5: class N ----------------
def fig5(out='S31_uniN.pdf'):
    fig, A = grid(2, 3, 1.80)
    A[0][0][1].remove()
    zs = zbar('standard'); z99 = zbar('U_N_0.01'); z90 = zbar('U_N_0.10'); z70 = zbar('U_N_0.30')
    picture(A[0][0][0], 'U_N_0.30', 23, '(a) N, $\\alpha_N=0.70$')
    r99, r70 = delta(z99, zs), delta(z70, zs); st = delta(z70, z90)
    ns, ew = mirror_ns(r70), mirror_ew(r70)
    dmap(fig, *A[0][1], r99, 201, lim_window(r99, 201), '(b) reference, $\\alpha_N=0.99$')
    dmap(fig, *A[0][2], r70, 201, lim_window(r70, 201), '(c) reference, $\\alpha_N=0.70$')
    dmap(fig, *A[1][0], st, 31, lim_nonzero(st, 31), '(d) step, $\\alpha_N$: 0.90 $\\rightarrow$ 0.70')
    dmap(fig, *A[1][1], ns, 23, lim_nonzero(ns, 23), '(e) N–S imbalance, $\\alpha_N=0.70$')
    dmap(fig, *A[1][2], ew, 23, lim_nonzero(ew, 23), '(f) E–W imbalance, $\\alpha_N=0.70$')
    fig.savefig(out); plt.close(fig)

# ------------- 4-column rows (Figs. 6 and 7) -------------
def four_rows(rows, out, s=1.30):
    fig, A = grid(len(rows), 4, s, between=0.52)
    zs = zbar('standard')
    for i, r in enumerate(rows):
        A[i][0][1].remove()
        L = 'abcdefghijkl'[4 * i:4 * i + 4]
        picture(A[i][0][0], r['key'], r['h'], '(%s) %s' % (L[0], r['title']),
                true_angle=r.get('true_angle', False), push=r.get('push', False))
        z = zbar(r['key']); st = delta(z, zbar(r['prev']))
        ref = delta(z, zs); ns, ew = mirror_ns(ref), mirror_ew(ref)
        dmap(fig, *A[i][1], st, r['hs'], lim_nonzero(st, r['hs']), '(%s) %s' % (L[1], r['step_title']))
        dmap(fig, *A[i][2], ns, r['h'], lim_nonzero(ns, r['h']), '(%s) N–S imbalance' % L[2])
        dmap(fig, *A[i][3], ew, r['h'], lim_nonzero(ew, r['h']), '(%s) E–W imbalance' % L[3])
    fig.savefig(out); plt.close(fig)

def fig6(out='S31_uniNS_NSE.pdf'):
    four_rows([
        dict(key='U_NS_0.30', prev='U_NS_0.10', h=16, hs=23, title='NS, $\\alpha_d=0.70$',
             step_title='step, 0.90 $\\rightarrow$ 0.70'),
        dict(key='U_NSE_0.10', prev='U_NSE_0.05', h=21, hs=27, title='NSE, $\\alpha_d=0.90$',
             step_title='step, 0.95 $\\rightarrow$ 0.90')], out)

def fig7(out='S31_nonuni.pdf'):
    four_rows([
        dict(key='F_N_0.01_S_0.05', prev='U_N_0.01', h=36, hs=63, push=True,
             title='$\\alpha_N=0.99$, $\\alpha_S=0.95$', step_title='step: S added'),
        dict(key='F_N_0.01_W_0.05', prev='U_N_0.01', h=36, hs=63, push=True,
             title='$\\alpha_N=0.99$, $\\alpha_W=0.95$', step_title='step: W added'),
        dict(key='F_N_0.01_S_0.05_E_0.10_W_0.30', prev='F_N_0.01_S_0.05_E_0.10', h=17, hs=27, push=True,
             true_angle=True, title='four directions', step_title='step: W added')], out)

# ---------------- Fig. 8: centre shift of all uniform classes ----------------
LV = ['0.01', '0.02', '0.05', '0.10', '0.30', '0.50']
AL = [0.99, 0.98, 0.95, 0.90, 0.70, 0.50]
def shift_of(key):
    g = geometry(zbar(key)); return math.hypot(g['dy'], g['dx'])
def fig8(out='S31_chart_shift.pdf'):
    fams = [('(a) single direction', ['N', 'S', 'E', 'W']),
            ('(b) neighbouring pairs', ['NE', 'NW', 'SE', 'SW']),
            ('(c) three directions', ['NSE', 'NSW', 'NEW', 'SEW']),
            ('(d) opposite pairs, all four', ['NS', 'EW', 'NSEW'])]
    cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    fig, axs = plt.subplots(1, 4, figsize=(TEXTW, 2.6), sharey=True)
    x = np.arange(len(AL))
    floor = max(shift_of('U_%s_%s' % (c, l)) for c in ['NS', 'EW', 'NSEW'] for l in LV)   # noise floor (0.64, quoted as 0.6)
    for ax, (t, cl) in zip(axs, fams):
        ax.axhspan(0, floor, color='#dddddd', zorder=0, lw=0)
        for c, col in zip(cl, cols):
            ax.plot(x, [shift_of('U_%s_%s' % (c, l)) for l in LV], '-o', color=col, ms=3.2, lw=1.2, label=c)
        ax.set_xticks(x); ax.set_xticklabels(['%.2f' % a for a in AL], rotation=90, fontsize=7)
        ax.set_xlabel('$\\alpha_d$', labelpad=1); ax.set_title(t, fontsize=8, pad=3)
        ax.grid(alpha=0.3, lw=0.5); ax.set_ylim(-0.1, 5.3)
        ax.legend(fontsize=7, loc='upper left', frameon=False, handlelength=1.4, borderaxespad=0.2)
        ax.tick_params(labelsize=7, length=2)
    axs[0].set_ylabel('centre shift (sites)')
    fig.subplots_adjust(left=0.065, right=0.995, top=0.91, bottom=0.22, wspace=0.08)
    fig.savefig(out); plt.close(fig)

# ---------------- Fig. 9: area vs total removal; shift vs net push ----------------
def fig9(out='S31_chart_both.pdf'):
    keys = sorted(f[:-8] for f in os.listdir(R) if f.endswith('__s0.npz') and f != 'standard__s0.npz')
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTW, 3.3), gridspec_kw=dict(width_ratios=[1.12, 1]))
    U = [k for k in keys if k.startswith('U_')]; F = [k for k in keys if k.startswith('F_')]
    lam = lambda k: sum(1 - v for v in alphas_of(k).values())
    area = lambda k: int((np.abs(zbar(k) - 2.0) > 1e-12).sum())
    LU = np.array([lam(k) for k in U]); AU = np.array([area(k) for k in U])
    LF = np.array([lam(k) for k in F]); AF = np.array([area(k) for k in F])
    slope, icpt = np.polyfit(np.log(np.r_[LU, LF]), np.log(np.r_[AU, AF]), 1)
    xx = np.array([0.008, 2.5])
    a1.plot(xx, np.exp(icpt) * xx ** slope, ls='--', color='#555555', lw=0.8, zorder=2,
            label='power-law fit, slope %.2f' % slope)
    a1.scatter(LU, AU, s=12, color='#708090', label='uniform regime (90 simulations)', zorder=3)
    a1.scatter(LF, AF, s=40, marker='*', color='#d01010', label='non-uniform regime (11 simulations)', zorder=4)
    a1.set_xscale('log'); a1.set_yscale('log')
    for ax in (a1, a2):                       # classic frame: full box, inward ticks on all four sides
        for side in ax.spines.values(): side.set_linewidth(0.7)
        ax.tick_params(which='both', direction='in', top=True, right=True, width=0.7)
        ax.tick_params(which='major', length=3.5); ax.tick_params(which='minor', length=2)
    a1.set_xlim(0.008, 2.5)
    a1.set_xlabel('total removal $\\lambda=\\sum_d(1-\\alpha_d)$'); a1.set_ylabel('area of the avalanche region (sites)')
    a1.set_title('(a) area against total removal', fontsize=8)
    lg = a1.legend(fontsize=7, frameon=True, fancybox=False, framealpha=1, loc='upper right', borderpad=0.5)
    lg.get_frame().set_edgecolor('#444444'); lg.get_frame().set_linewidth(0.6); lg.set_zorder(10); a1.tick_params(labelsize=7)
    special = {'F_N_0.01_W_0.05': ('#e08000', '$\\alpha_N=0.99$, $\\alpha_W=0.95$ (11$^\\circ$ off)'),
               'F_S_0.01_E_0.05_W_0.10': ('#d01010', '$\\alpha_S=0.99$, $\\alpha_E=0.95$, $\\alpha_W=0.90$ (18$^\\circ$ off)')}
    for k in F:
        g = geometry(zbar(k)); mx, my = g['dx'], -g['dy']
        vx, vy = net_push(k); vy = -vy; n = math.hypot(vx, vy)
        col = special[k][0] if k in special else '#0050c8'
        a2.plot([0, 3.9 * vx / n], [0, 3.9 * vy / n], ls=(0, (1, 1.6)), color='#00a050', lw=0.9, zorder=2)
        a2.annotate('', xy=(mx, my), xytext=(0, 0), zorder=4 if k in special else 3,
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=1.6 if k in special else 1.3,
                                    mutation_scale=9, shrinkA=0, shrinkB=0))
    a2.plot([], [], ls=(0, (1, 1.6)), color='#00a050', lw=1.2, label='net push direction')
    a2.plot([], [], color='#0050c8', lw=1.4, label='shift within 5$^\\circ$ of the push')
    for k, (col, lab) in special.items():
        a2.plot([], [], color=col, lw=1.6, label='shift, ' + lab)
    a2.axhline(0, color='k', lw=0.5); a2.axvline(0, color='k', lw=0.5)
    a2.set_xlim(-5.2, 5.2); a2.set_ylim(-4, 6.4); a2.set_aspect('equal'); a2.set_yticks(range(-4, 5, 2)); a2.set_xticks(range(-4, 5, 2))
    a2.set_xlabel('east–west shift (sites)'); a2.set_ylabel('north–south shift (sites)')
    a2.set_title('(b) non-uniform regime: 11 configurations', fontsize=8)
    a2.tick_params(labelsize=7)
    lg2 = a2.legend(fontsize=6, frameon=True, fancybox=False, framealpha=1, loc='upper center', borderpad=0.45,
                    handlelength=1.6, labelspacing=0.3, borderaxespad=0.3)
    lg2.get_frame().set_edgecolor('#444444'); lg2.get_frame().set_linewidth(0.6); lg2.set_zorder(10)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.93, bottom=0.15, wspace=0.22)
    fig.savefig(out); plt.close(fig)



def fig1(out='FIG1.png', h=183, px=6):
    """Fig. 1: final configuration of the standard sandpile, cropped around the injection site."""
    fin = np.load(R + 'standard__s0.npz')['final'].astype(int)
    crop = PAL[np.clip(win(fin, h), 0, 3)]
    img = Image.fromarray(np.kron(crop, np.ones((px, px, 1), np.uint8)).astype(np.uint8))
    W = img.width; band = 110
    canvas = Image.new('RGB', (W, W + band), 'white'); canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas); f = ImageFont.truetype(font_manager.findfont('DejaVu Sans:bold'), 44)
    for k in range(4):
        x0 = 40 + k * W // 4; y0 = W + 25
        d.rectangle([x0, y0, x0 + 55, y0 + 55], fill=tuple(int(v) for v in PAL[k]), outline=(60, 60, 60), width=2)
        d.text((x0 + 70, y0 + 28), 'z = %d' % k, font=f, fill=(34, 34, 34), anchor='lm')
    canvas.save(out)

# ---------------- Figs. 3 and 4: picture panels ----------------
OUT = 'figures_section3'
PGREEN=(0,255,136); PX=600
font=ImageFont.truetype(font_manager.findfont('DejaVu Sans'),44)
def reach_of(key):
    z=np.load(R+key+'__s0.npz')['zbar']; rr,cc=np.nonzero(np.abs(z-2.0)>1e-12)
    return max(C0-rr.min(), rr.max()-C0, cc.max()-C0, C0-cc.min())
def render(key,name,h):
    d=np.load(R+key+'__s0.npz'); z=d['zbar']; f=d['final'].astype(int)
    reg=np.abs(z-2.0)>1e-12                      # sites reached by at least one avalanche
    rr,cc=np.nonzero(reg)
    crop=f[C0-h:C0+h+1, C0-h:C0+h+1]
    n=crop.shape[0]; s=PX/n
    im=Image.fromarray(PAL[np.clip(crop,0,3)]).resize((PX,PX),Image.NEAREST)
    dr=ImageDraw.Draw(im)
    cx=cy=(h+0.5)*s                              # injection site (pixel centre)
    dash,gap=16,10
    for t in range(0,PX,dash+gap):               # dashed cross through the injection site
        dr.line([(t,cy),(min(t+dash,PX),cy)],fill='white',width=5)
        dr.line([(cx,t),(cx,min(t+dash,PX))],fill='white',width=5)
    gx=(cc.mean()-(C0-h)+0.5)*s; gy=(rr.mean()-(C0-h)+0.5)*s   # centre of the avalanche region
    r=24; dr.ellipse([gx-r,gy-r,gx+r,gy+r],fill=PGREEN,outline='black',width=5)
    BAND=70                                       # white strip below the picture for the 10-site scale bar
    out=Image.new('RGB',(PX,PX+BAND),'white'); out.paste(im,(0,0)); d2=ImageDraw.Draw(out)
    L=10*s; x0=6; yb=PX+BAND//2
    d2.rectangle([x0,yb-5,x0+L,yb+5],fill='black')
    d2.text((x0+L+14,yb),'10 sites',font=font,fill='black',anchor='lm')
    out.save(os.path.join(OUT, name))
    return round(rr.mean()-C0,2), round(cc.mean()-C0,2)

def panels():
    """Figs. 3 and 4: space-time picture panels M_*.png (one zoom per row)."""
    fig3=[('U_N_0.01','A_N_001'),('U_N_0.02','A_N_002'),('U_N_0.05','A_N_005'),('U_N_0.10','A_N_01'),
          ('U_S_0.01','A_S_001'),('U_E_0.01','A_E_001'),('U_W_0.01','A_W_001'),
          ('U_N_0.30','X_N_03'),('U_S_0.30','X_S_03'),('U_E_0.30','X_E_03'),('U_W_0.30','X_W_03'),
          ('U_NS_0.30','X_NS_03'),('U_NE_0.30','X_NE_03'),('U_NW_0.30','X_NW_03'),('U_SE_0.30','X_SE_03'),
          ('U_SW_0.30','X_SW_03'),('U_EW_0.30','X_EW_03'),('U_NSE_0.10','X_NSE_01'),('U_NSW_0.10','X_NSW_01'),
          ('U_NEW_0.10','X_NEW_01'),('U_SEW_0.10','X_SEW_01'),('U_NSEW_0.30','X_NSEW_03')]
    fig4=[('F_N_0.01_S_0.05','Y_N01_S05'),('F_N_0.01_E_0.05','Y_N01_E05'),('F_N_0.01_W_0.05','Y_N01_W05'),
          ('F_S_0.01_E_0.05','Y_S01_E05'),('F_S_0.01_W_0.05','Y_S01_W05'),('F_E_0.01_W_0.05','Y_E01_W05'),
          ('F_N_0.01_S_0.05_E_0.10','Y_N01_S05_E1'),('F_N_0.01_S_0.05_W_0.10','Y_N01_S05_W1'),
          ('F_E_0.01_W_0.05_N_0.10','Y_E01_W05_N1'),('F_S_0.01_E_0.05_W_0.10','Y_S01_E05_W1'),
          ('F_N_0.01_S_0.05_E_0.10_W_0.30','Y_N01_S05_E1_W3')]
    d3=dict((n,(k,n)) for k,n in fig3)
    rows3=[fig3[0:4],None,fig3[7:11],[d3[x] for x in ['X_NE_03','X_NW_03','X_SE_03','X_SW_03']],[d3[x] for x in ['X_NS_03','X_EW_03','X_NSEW_03']],[d3[x] for x in ['X_NSE_01','X_NSW_01','X_NEW_01','X_SEW_01']]]
    rows3[1]=[('U_N_0.01','A_N_001')]+fig3[4:7]   # row 2 of Fig. 3 repeats N at 0.99
    rows4=[fig4[0:4],fig4[4:8],fig4[8:11]]
    for row in rows3+rows4:
        h=int(math.ceil(1.25*max(reach_of(k) for k,_ in row)))+2   # one zoom per row, centred on the injection site
        for k,n in row: render(k,'M_'+n+'.png',h)
        print([n for _,n in row],'half-width',h)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='figures_section3')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    OUT = a.out
    J = lambda f: os.path.join(a.out, f)
    fig1(J('FIG1.png'))
    panels()
    fig5(J('S31_uniN.pdf')); fig6(J('S31_uniNS_NSE.pdf')); fig7(J('S31_nonuni.pdf'))
    fig8(J('S31_chart_shift.pdf')); fig9(J('S31_chart_both.pdf'))
    print('figures written to', a.out)
