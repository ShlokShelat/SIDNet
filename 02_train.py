"""
Train SIDNet with the five-phase curriculum of Sec. 6 (Table 6).

  Phase  classes                              epochs U/N  LR      b_cls  b_dir, b_alp U / N
  A      all 16 (no class loss)               25 / 40     3e-4    0      3.0, 2.0 / 4.0, 3.0
  B1     standard only                        20 / 20     3e-4    1      2.0, 1.0
  B2     + N, S, E, W, NS, EW                 30 / 30     2e-4    1      2.0, 1.0 / 2.5, 1.5
  B3     + NE, NW, SE, SW                     30 / 40     1e-4    1      2.0, 1.0 / 2.5, 1.5
  B4     + three-direction, NSEW              40 / 60     1e-4    1      2.0, 1.0 / 2.5, 1.5

AdamW (weight decay 1e-4), cosine schedule to 5 % of the starting LR within each
phase, batch size 32, gradient accumulation over 4 steps in B3 and B4 (effective
batch 128), gradient clipping to an l2 norm of 1.0, class-balanced sampler.

Model selection on the validation set after every epoch:
  A      mean per-direction F1 (s_hat_d > 0.5), averaged over N, S, E, W
  B1     lowest total validation loss
  B2-B4  classification accuracy over the classes introduced so far
The best model of each phase starts the next one; the final model is the best of B4.

The full training state is saved after every epoch (<regime>_last.pt). If the job
stops, running the same command again continues from the last finished epoch.
The validation history of each phase is written to <regime>_phase<P>_history.json.

Usage
  python 02_train.py --regime uni  --data_dir data --out_dir checkpoints
  python 02_train.py --regime nuni --data_dir data --out_dir checkpoints
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sid_common import N_DIRS, NOISE_NAMES
from sid_data import SIDDataset, class_balanced_sampler
from sid_losses import class_weights, direction_targets, total_loss
from sidnet import SIDNet, count_parameters

SINGLE = ["N", "S", "E", "W"]
PHASE_CLASSES = {
    "A": NOISE_NAMES,
    "B1": ["standard"],
    "B2": ["standard"] + SINGLE + ["NS", "EW"],
    "B3": ["standard"] + SINGLE + ["NS", "EW", "NE", "NW", "SE", "SW"],
    "B4": NOISE_NAMES,
}
#            epochs(U, N)  LR     accum  b_cls  (b_dir, b_alp) U       (b_dir, b_alp) N
PHASE_CFG = {
    "A":  ((25, 40), 3e-4, 1, 0.0, (3.0, 2.0), (4.0, 3.0)),
    "B1": ((20, 20), 3e-4, 1, 1.0, (2.0, 1.0), (2.0, 1.0)),
    "B2": ((30, 30), 2e-4, 1, 1.0, (2.0, 1.0), (2.5, 1.5)),
    "B3": ((30, 40), 1e-4, 4, 1.0, (2.0, 1.0), (2.5, 1.5)),
    "B4": ((40, 60), 1e-4, 4, 1.0, (2.0, 1.0), (2.5, 1.5)),
}
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
FINAL_LR_FRACTION = 0.05
CLIP_NORM = 1.0


def phase_settings(phase, regime):
    (ep_u, ep_n), lr, accum, b_cls, w_u, w_n = PHASE_CFG[phase]
    b_dir, b_alp = w_u if regime == "uni" else w_n
    return (ep_u if regime == "uni" else ep_n), lr, accum, b_cls, b_dir, b_alp


PHASE_ORDER = ["A", "B1", "B2", "B3", "B4"]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def epoch_seed(seed, phase, ep):
    """Seed of one epoch. Sampling order and dropout depend only on (seed, phase, epoch), so a run
    that is interrupted and resumed follows the same sequence as an uninterrupted run."""
    return seed * 1_000_003 + PHASE_ORDER.index(phase) * 1_000 + ep


def run_info(args, device, model):
    """Record of the software, hardware and settings of a training run."""
    import platform, subprocess, h5py
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or None
    except OSError:
        commit = None
    return {"args": vars(args), "git_commit": commit, "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "h5py": h5py.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "parameters": count_parameters(model), "started": time.strftime("%Y-%m-%d %H:%M:%S")}


@torch.no_grad()
def validate(model, loader, device, w, b_cls, b_dir, b_alp):
    model.eval()
    n = correct = 0
    loss_sum = 0.0
    tp = torch.zeros(N_DIRS); fp = torch.zeros(N_DIRS); fn = torch.zeros(N_DIRS)
    for delta, ws, wl, y, alpha in loader:
        delta, ws, wl, y, alpha = (t.to(device, non_blocking=True) for t in (delta, ws, wl, y, alpha))
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits, s_hat, a_hat = model(delta, ws, wl)
        loss, _ = total_loss(logits, s_hat, a_hat, y, alpha, w, b_cls, b_dir, b_alp)
        loss_sum += loss.item() * y.numel()
        correct += (logits.argmax(-1) == y).sum().item()
        n += y.numel()
        pred = (s_hat.float() > 0.5).cpu(); true = direction_targets(y).bool().cpu()
        tp += (pred & true).sum(0); fp += (pred & ~true).sum(0); fn += (~pred & true).sum(0)
    f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    return {"loss": loss_sum / max(n, 1), "acc": correct / max(n, 1), "dir_f1": f1.mean().item()}


def train_phase(phase, model, regime, args, device, resume=None):
    """Train one phase. `resume` is the last-epoch state of this phase, if the job was interrupted."""
    n_ep, lr, accum, b_cls, b_dir, b_alp = phase_settings(phase, regime)
    if args.quick:
        n_ep = 1
    batch = 8 if args.quick else BATCH_SIZE
    classes = PHASE_CLASSES[phase]
    print(f"\n=== [{regime}] Phase {phase}: {len(classes)} classes, {n_ep} epochs, lr {lr:g}, "
          f"accum {accum}, b_cls {b_cls}, b_dir {b_dir}, b_alp {b_alp}", flush=True)

    # Phase A: the classification loss is off and the classification head is not trained
    for p in model.head.parameters():
        p.requires_grad = phase != "A"

    train_ds = SIDDataset(args.data_dir, regime, "train", classes)
    val_ds = SIDDataset(args.data_dir, regime, "val", classes)
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.workers > 0)
    sampler_gen = torch.Generator()
    train_dl = DataLoader(train_ds, batch, sampler=class_balanced_sampler(train_ds, sampler_gen),
                          drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, batch, shuffle=False, **loader_kw)

    w = class_weights().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep, eta_min=lr * FINAL_LR_FRACTION)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_path = os.path.join(args.out_dir, f"{regime}_phase{phase}_best.pt")
    last_path = os.path.join(args.out_dir, f"{regime}_last.pt")
    hist_path = os.path.join(args.out_dir, f"{regime}_phase{phase}_history.json")
    best, history, start = None, [], 1
    if resume is not None:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        sched.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        best, history, start = resume["best"], resume["history"], resume["epoch"] + 1
        print(f"  resuming Phase {phase} at epoch {start}", flush=True)

    for ep in range(start, n_ep + 1):
        model.train()
        sampler_gen.manual_seed(epoch_seed(args.seed, phase, ep))
        torch.manual_seed(epoch_seed(args.seed, phase, ep))
        t0, run_loss, seen, step = time.time(), 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for step, (delta, ws, wl, y, alpha) in enumerate(train_dl, 1):
            delta, ws, wl, y, alpha = (t.to(device, non_blocking=True) for t in (delta, ws, wl, y, alpha))
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits, s_hat, a_hat = model(delta, ws, wl)
            loss, _ = total_loss(logits, s_hat, a_hat, y, alpha, w, b_cls, b_dir, b_alp)
            scaler.scale(loss / accum).backward()
            if step % accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, CLIP_NORM)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
            run_loss += loss.item() * y.numel(); seen += y.numel()
            if args.log_every and step % args.log_every == 0:
                print(f"    step {step:6d}/{len(train_dl)}  loss {run_loss / seen:.4f}", flush=True)
            if args.quick and step >= 10:
                break
        if step % accum:                                  # apply the gradients of a last partial accumulation
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, CLIP_NORM)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
        sched.step()

        m = validate(model, val_dl, device, w, b_cls, b_dir, b_alp)
        if phase == "A":
            score, better = m["dir_f1"], best is None or m["dir_f1"] > best
        elif phase == "B1":
            score, better = m["loss"], best is None or m["loss"] < best
        else:
            score, better = m["acc"], best is None or m["acc"] > best
        lr_now = opt.param_groups[0]["lr"]
        print(f"  ep {ep:3d}/{n_ep}  train_loss {run_loss / max(seen, 1):.4f}  val_loss {m['loss']:.4f}  "
              f"val_acc {100 * m['acc']:.2f}%  val_dir_F1 {100 * m['dir_f1']:.2f}%  "
              f"lr {lr_now:.2e}  {time.time() - t0:.0f}s", flush=True)
        history.append({"epoch": ep, "train_loss": run_loss / max(seen, 1), "lr": lr_now, **m})
        if better:
            best = score
            torch.save({"model": model.state_dict(), "phase": phase, "epoch": ep,
                        "regime": regime, "val": m}, best_path)
            print(f"    saved best ({'F1' if phase == 'A' else 'loss' if phase == 'B1' else 'acc'} "
                  f"{score:.4f}) -> {best_path}", flush=True)
        # state for resuming after an interruption (written every epoch, atomically)
        torch.save({"phase": phase, "epoch": ep, "model": model.state_dict(),
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict(), "best": best, "history": history},
                   last_path + ".tmp")
        os.replace(last_path + ".tmp", last_path)
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=1)

    model.load_state_dict(torch.load(best_path, map_location=device)["model"])  # best model starts the next phase
    print(f"  Phase {phase} done: best model of epoch "
          f"{torch.load(best_path, map_location='cpu')['epoch']} -> {best_path}", flush=True)
    return best_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regime", required=True, choices=["uni", "nuni"])
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--phases", default="A,B1,B2,B3,B4")
    p.add_argument("--no_resume", action="store_true",
                   help="start from scratch even if an interrupted run is found in out_dir")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=500, help="print the running loss every N steps (0: off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true",
                   help="deterministic GPU kernels (slower); without it, cuDNN autotuning and TF32 are used")
    p.add_argument("--quick", action="store_true",
                   help="pipeline test: 1 epoch per phase, 10 steps per epoch, batch 8")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        fast = not a.deterministic
        torch.backends.cuda.matmul.allow_tf32 = fast
        torch.backends.cudnn.allow_tf32 = fast
        torch.backends.cudnn.benchmark = fast
        torch.backends.cudnn.deterministic = a.deterministic
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: no GPU found, training on CPU will be very slow")
    model = SIDNet().to(device)
    print(f"SIDNet: {count_parameters(model):,} trainable parameters")
    info_path = os.path.join(a.out_dir, f"{a.regime}_run_info.json")
    with open(info_path, "a") as f:                       # one line per start or resume of the run
        f.write(json.dumps(run_info(a, device, model)) + "\n")

    phases = [s.strip() for s in a.phases.split(",") if s.strip()]
    last_path = os.path.join(a.out_dir, f"{a.regime}_last.pt")
    resume = None
    if not a.no_resume and os.path.exists(last_path):
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["phase"] in phases:
            k = phases.index(state["phase"])
            n_ep = 1 if a.quick else phase_settings(state["phase"], a.regime)[0]
            if state["epoch"] >= n_ep:                      # that phase had finished
                model.load_state_dict(torch.load(os.path.join(
                    a.out_dir, f"{a.regime}_phase{state['phase']}_best.pt"), map_location=device)["model"])
                phases = phases[k + 1:]
                print(f"resuming after completed Phase {state['phase']}")
            else:
                phases, resume = phases[k:], state
                print(f"resuming Phase {state['phase']} after epoch {state['epoch']}")

    for i, phase in enumerate(phases):
        train_phase(phase, model, a.regime, a, device, resume=resume if i == 0 else None)
    print(f"\nTraining complete. Final model: {os.path.join(a.out_dir, f'{a.regime}_phaseB4_best.pt')}")


if __name__ == "__main__":
    main()
