#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Napari batch viewer + scatter CSV exporter (UNMASKED + ColocZ masked, CUDA optional)
Masks are configurable: Otsu, Percentile, or Hybrid (max(Otsu, Percentile)) per channel.

Per file it:
  • Displays Ch1/Ch2 in Napari (1–99% contrast; magenta/green)
  • Shows masks (QC) and a WHITE ColocZ AND-overlay
  • Saves a trimmed Napari screenshot: <stem>_napari_overview.png
  • Writes TWO CSVs:
      <stem>_scatter_points_unmasked.csv      # ch1,ch2 (all finite pixels)
      <stem>_scatter_points_ColocZ.csv        # ch1,ch2 (Ch1>thr1 AND Ch2>thr2)
  • Writes TWO scatter previews:
      <stem>_intensity_scatter.png
      <stem>_intensity_scatter_ColocZ.png

Default mask setup
------------------
* --mask-mode hybrid  → threshold = max(Otsu, Percentile)
* Percentile targets start at 50% but are clamped to a window **[35–70]**:
    --p1 50 --p2 50 --pmin 35 --pmax 70

Red-channel inclusivity
-----------------------
* Use --red-inclusive-frac (default 0.40) to make **Ch1** more inclusive:
    - Percentile mode: p1 is pulled down toward pmin by this fraction
    - Otsu/Hybrid: Ch1 Otsu intensity is pulled down toward the Ch1 pmin-intensity
    - Ch2 (green) is unchanged
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ---------- Optional CUDA backend ----------
xp = np
cp = None
def try_cuda(enable: bool) -> bool:
    global xp, cp
    if not enable:
        return False
    try:
        import cupy as cp_
        if cp_.cuda.runtime.getDeviceCount() <= 0:
            return False
        cp = cp_
        xp = cp_
        return True
    except Exception:
        return False

def to_cpu(a):
    if cp is not None:
        try:
            import cupy as _cp
            if isinstance(a, _cp.ndarray):
                return _cp.asnumpy(a)
        except Exception:
            pass
    return a

def to_xp(a):
    if cp is not None:
        try:
            import cupy as _cp
            return _cp.asarray(a)
        except Exception:
            pass
    return np.asarray(a)

# ---------- TIFF loader ----------
try:
    import tifffile as tiff
except Exception:
    tiff = None

def load_image(path: str) -> np.ndarray:
    if tiff is None:
        raise RuntimeError("Please install 'tifffile' (pip install tifffile).")
    arr = tiff.imread(path)
    return np.asarray(arr, dtype=np.float32)

def to_CZYX(a: np.ndarray) -> np.ndarray:
    """Return array shaped (C, Z, Y, X). Z=1 for 2D images."""
    a = np.asarray(a)
    if a.ndim == 2:                      # (Y,X)
        return a[np.newaxis, np.newaxis, :, :]
    if a.ndim == 3:
        if a.shape[0] <= 4:              # (C,Y,X)
            return a[:, np.newaxis, :, :]
        if a.shape[-1] <= 4:             # (Y,X,C)
            return a.transpose(2, 0, 1)[:, np.newaxis, :, :]
        return a[np.newaxis, ...]        # (Z,Y,X)
    if a.ndim == 4:
        if a.shape[-1] <= 4:             # (Z,Y,X,C)
            return a.transpose(3, 0, 1, 2)
        if a.shape[0] <= 4:              # (C,Z,Y,X)
            return a
        return a.transpose(3, 0, 1, 2)   # assume (Z,Y,X,C)
    raise ValueError(f"Unsupported image shape: {a.shape}")

# ---------- Otsu (CPU) ----------
def otsu_threshold_cpu(v: np.ndarray) -> float:
    """Return Otsu threshold on CPU (safe even if CUDA is on)."""
    from skimage.filters import threshold_otsu
    v = np.asarray(v, dtype=np.float32)
    v = v[np.isfinite(v)]
    if v.size == 0 or np.all(v == 0):
        return 0.0
    try:
        return float(threshold_otsu(v))
    except Exception:
        return 0.0

# ---------- Percentile helpers ----------
def percentile_value(v: np.ndarray, pct: float) -> float:
    """Percentile value (0–100) on finite data; returns 0.0 if degenerate."""
    v = np.asarray(v, dtype=np.float32)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    try:
        return float(np.percentile(v, float(pct)))
    except Exception:
        return 0.0

# ---------- CSV + PNG writers ----------
def write_pairs_csv(a_cpu: np.ndarray, b_cpu: np.ndarray, out_csv: Path, chunk: int = 5_000_000) -> int:
    """Stream ch1,ch2 pairs to CSV. Returns number of rows."""
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    good = np.isfinite(a_cpu) & np.isfinite(b_cpu)
    a = a_cpu[good]; b = b_cpu[good]

    with open(out_csv, "w", newline="") as f:
        f.write("ch1,ch2\n")
        n = a.size
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            block = np.column_stack((a[start:end], b[start:end]))
            np.savetxt(f, block, delimiter=",", fmt="%.6g")
    return a.size

def _scatter_or_hexbin(ax, x, y, hexbin: bool, subsample: int):
    if subsample and x.size > subsample:
        rng = np.random.RandomState(0)
        idx = rng.choice(x.size, size=subsample, replace=False)
        x = x[idx]; y = y[idx]
    if hexbin:
        ax.hexbin(x, y, gridsize=200, bins='log')
    else:
        ax.scatter(x, y, s=1, alpha=0.10, linewidths=0)
    return x, y

def save_scatter_png(a_cpu: np.ndarray, b_cpu: np.ndarray, out_png: Path,
                     title: str, hexbin: bool = False, preview_max: int = 0,
                     draw_fit: bool = True, ref_mode: str = "none"):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    good = np.isfinite(a_cpu) & np.isfinite(b_cpu)
    x = a_cpu[good]; y = b_cpu[good]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    xp_, yp_ = _scatter_or_hexbin(ax, x, y, hexbin=hexbin, subsample=int(preview_max))

    if xp_.size > 1:
        xline = np.linspace(float(np.min(xp_)), float(np.max(xp_)), 200)
        if draw_fit:
            m, b = np.polyfit(xp_, yp_, 1)
            ax.plot(xline, m * xline + b, color="black", lw=2.0, label="preview fit")
        ref_mode = (ref_mode or "none").lower()
        if ref_mode != "none":
            if ref_mode == "unit":
                k = 1.0
            elif ref_mode.startswith("gain:"):
                try:
                    k = float(ref_mode.split(":", 1)[1])
                except Exception:
                    k = np.nan
            else:
                k = np.nan
            if np.isfinite(k):
                ax.plot(xline, k * xline, color="black", lw=2.0, ls="--", label=f"ref: y={k:g}x")
        if draw_fit or ref_mode != "none":
            ax.legend(loc="upper left", frameon=True)

    ax.set_xlabel("Channel 1 intensity")
    ax.set_ylabel("Channel 2 intensity")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ---------- GUI helpers ----------
def choose_folder_gui(title="Select folder with TIFFs") -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        d = filedialog.askdirectory(title=title)
        return Path(d) if d else None
    except Exception:
        return None

# ---------- Mask utilities ----------
def clamp_percentiles(p: float, pmin: float, pmax: float) -> float:
    pmin = float(np.clip(pmin, 0.0, 100.0))
    pmax = float(np.clip(pmax, 0.0, 100.0))
    if pmin > pmax:
        pmin, pmax = pmax, pmin
    return float(np.clip(p, pmin, pmax))

def make_masks(ch1: np.ndarray, ch2: np.ndarray,
               mode: str,
               p1: float, p2: float, pmin: float, pmax: float,
               red_inc_frac: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                            float, float, float, float,
                                            float, float, float, float]:
    """
    Returns (m1, m2, mand,
             T1o, T2o, T1p_eff, T2p,
             p1_used_eff, p2_used, T1o_eff, T1p_raw)

    mode in {"otsu","percentile","hybrid"}.
    red_inc_frac in [0, 1] pulls Ch1 thresholds down (more inclusive).
    """
    # Clamp target percentiles and compute raw percentile intensities
    p1u = clamp_percentiles(p1, pmin, pmax)
    p2u = clamp_percentiles(p2, pmin, pmax)
    # pmin intensities (for “baseline” when making Otsu more inclusive)
    ch1_pmin_val = percentile_value(ch1, pmin)
    ch2_pmin_val = percentile_value(ch2, pmin)

    # Raw percentile thresholds
    T1p_raw = percentile_value(ch1, p1u)
    T2p = percentile_value(ch2, p2u)

    # Effective Ch1 percentile after inclusivity: pull toward pmin by red_inc_frac
    #   p1_eff = pmin + (p1u - pmin) * (1 - red_inc_frac)
    p1_eff = pmin + (p1u - pmin) * max(0.0, 1.0 - float(red_inc_frac))
    p1_eff = clamp_percentiles(p1_eff, pmin, pmax)
    T1p_eff = percentile_value(ch1, p1_eff)

    # Otsu thresholds
    T1o = otsu_threshold_cpu(ch1)
    T2o = otsu_threshold_cpu(ch2)

    # Make Otsu more inclusive for Ch1 by pulling it toward the pmin intensity
    #   T1o_eff = ch1_pmin_val + (T1o - ch1_pmin_val) * (1 - red_inc_frac)
    T1o_eff = ch1_pmin_val + (T1o - ch1_pmin_val) * max(0.0, 1.0 - float(red_inc_frac))

    mode = (mode or "otsu").lower()
    if mode == "percentile":
        T1 = T1p_eff
        T2 = T2p
    elif mode == "hybrid":
        # Hybrid = max( Otsu_eff, Percentile_eff ) for Ch1; standard max for Ch2
        T1 = max(T1o_eff, T1p_eff)
        T2 = max(T2o, T2p)
    else:  # otsu
        T1 = T1o_eff
        T2 = T2o

    m1 = (ch1 > T1).astype(np.uint8)
    m2 = (ch2 > T2).astype(np.uint8)
    mand = ((ch1 > T1) & (ch2 > T2)).astype(np.uint8)

    return (m1, m2, mand,
            T1o, T2o, T1p_eff, T2p,
            p1_eff, p2u, T1o_eff, T1p_raw)

# ---------- Core per-file processing ----------
def export_scatter_pairs_and_previews(CZYX: np.ndarray, ch1: int, ch2: int, stem: str, out_dir: Path,
                                      do_hexbin: bool, preview_max: int,
                                      draw_fit: bool, ref_mode: str,
                                      png_masked_only: bool,
                                      mask_mode: str, p1: float, p2: float, pmin: float, pmax: float,
                                      red_inc_frac: float) -> Tuple[int, int]:
    """
    Writes:
      stem_scatter_points_unmasked.csv
      stem_scatter_points_ColocZ.csv
      stem_intensity_scatter.png            (unless png_masked_only=True)
      stem_intensity_scatter_ColocZ.png
    Returns (n_unmasked, n_masked).
    """
    A = to_cpu(CZYX[ch1])
    B = to_cpu(CZYX[ch2])

    # --- Unmasked ---
    Au = A.reshape(-1).astype(np.float32, copy=False)
    Bu = B.reshape(-1).astype(np.float32, copy=False)

    csv_unmasked = out_dir / f"{stem}_scatter_points_unmasked.csv"
    n_unmasked = write_pairs_csv(Au, Bu, csv_unmasked)

    if not png_masked_only:
        png_unmasked = out_dir / f"{stem}_intensity_scatter.png"
        save_scatter_png(Au, Bu, png_unmasked,
                         title=f"Intensity Scatter — {stem} (UNMASKED)",
                         hexbin=do_hexbin, preview_max=preview_max,
                         draw_fit=draw_fit, ref_mode=ref_mode)

    # --- ColocZ mask (configurable thresholds; red channel more inclusive) ---
    (m1, m2, mand,
     T1o, T2o, T1p_eff, T2p,
     p1_eff, p2_used, T1o_eff, T1p_raw) = make_masks(
        A, B, mask_mode, p1, p2, pmin, pmax, red_inc_frac
    )

    Am = A[mand > 0].astype(np.float32, copy=False).ravel()
    Bm = B[mand > 0].astype(np.float32, copy=False).ravel()

    csv_masked = out_dir / f"{stem}_scatter_points_ColocZ.csv"
    n_masked = write_pairs_csv(Am, Bm, csv_masked)

    # Title with concise threshold info
    if mask_mode.lower() == "percentile":
        info = f"pct(Ch1:{p1_eff:.0f}%→{T1p_eff:.2g}, Ch2:{p2_used:.0f}%→{T2p:.2g})"
    elif mask_mode.lower() == "hybrid":
        info = (f"hybrid max[ "
                f"Ch1: Otsu_eff({T1o_eff:.2g}) vs p{p1_eff:.0f}%({T1p_eff:.2g}); "
                f"Ch2: Otsu({T2o:.2g}) vs p{p2_used:.0f}%({T2p:.2g}) ]")
    else:
        info = f"Otsu (Ch1_eff:{T1o_eff:.2g}, Ch2:{T2o:.2g})"

    png_masked = out_dir / f"{stem}_intensity_scatter_ColocZ.png"
    save_scatter_png(Am, Bm, png_masked,
                     title=f"Intensity Scatter — {stem} (ColocZ: {info})",
                     hexbin=do_hexbin, preview_max=preview_max,
                     draw_fit=draw_fit, ref_mode=ref_mode)

    return n_unmasked, n_masked

# ---------- Batch runner (Napari + QTimer) ----------
def run_with_viewer(args):
    import napari
    from magicgui import magicgui
    from magicgui.widgets import Label
    from qtpy.QtCore import QTimer, QEventLoop
    from qtpy.QtWidgets import QApplication

    used_cuda = try_cuda(args.cuda)
    in_dir = Path(args.folder) if args.folder else choose_folder_gui()
    if not in_dir:
        print("No folder selected. Exiting.")
        return
    out_dir = Path(args.out) if args.out else (in_dir / "scatter_exports")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(in_dir.glob(args.pattern)))
    if not files:
        print(f"No files matching {args.pattern} in {in_dir}")
        return

    # maximize viewer before screenshots
    viewer = napari.Viewer(title=f"Batch scatter export — CUDA={used_cuda}")
    app = QApplication.instance()
    if app is not None:
        viewer.window._qt_window.showMaximized()
        viewer.window._qt_window.raise_()
        viewer.window._qt_window.activateWindow()

    status = Label(value="Ready.")
    viewer.window.add_dock_widget(status, area="bottom")

    @magicgui(
        pause_ms={"label":"Pause per file (ms)","widget_type":"SpinBox","min":0,"max":10000,"step":50},
        stop_now={"label":"Stop after current","widget_type":"CheckBox"},
        call_button="Update"
    )
    def controls(pause_ms: int = args.pause_ms, stop_now: bool = False):
        pass
    viewer.window.add_dock_widget(controls, area="right")

    idx = {"i": 0}
    ch1 = max(0, int(args.ch1))
    ch2 = max(0, int(args.ch2))

    def add_layers(CZYX: np.ndarray, stem: str):
        viewer.layers.clear()

        def _clims(a):
            v = a[np.isfinite(a)]
            if v.size == 0:
                return (0.0, 1.0)
            lo = float(np.percentile(v, 1))
            hi = float(np.percentile(v, 99))
            if hi <= lo: hi = lo + 1e-6
            return (lo, hi)

        ch1_arr = to_cpu(CZYX[ch1])
        ch2_arr = to_cpu(CZYX[ch2])

        viewer.add_image(ch1_arr, name="Ch1", colormap="magenta",
                         blending="additive", contrast_limits=_clims(ch1_arr))
        viewer.add_image(ch2_arr, name="Ch2", colormap="green",
                         blending="additive", contrast_limits=_clims(ch2_arr))

        (m1, m2, mand,
         T1o, T2o, T1p_eff, T2p,
         p1_eff, p2_used, T1o_eff, T1p_raw) = make_masks(
            ch1_arr, ch2_arr, args.mask_mode, args.p1, args.p2, args.pmin, args.pmax, args.red_inclusive_frac
        )

        colors_ch1 = {0:(0,0,0,0), 1:(1,0,1,0.55)}
        colors_ch2 = {0:(0,0,0,0), 1:(0,1,0,0.55)}
        colors_and = {0:(0,0,0,0), 1:(1,1,1,0.60)}  # white coloc

        viewer.add_labels(m1, name="Mask Ch1 (RED, inclusive)", color=colors_ch1)
        viewer.add_labels(m2, name="Mask Ch2 (GREEN)", color=colors_ch2)
        L_and = viewer.add_labels(mand, name="ColocZ", color=colors_and)
        try:
            viewer.layers.move(viewer.layers.index(L_and), len(viewer.layers)-1)
        except Exception:
            pass

        viewer.text_overlay.visible = True
        viewer.text_overlay.text = (
            f"{stem}\n"
            f"Mode={args.mask_mode} | "
            f"Ch1: Otsu_eff={T1o_eff:.3g}, p_eff={p1_eff:.0f}%→{T1p_eff:.3g} "
            f"(inc={args.red_inclusive_frac:.0%}) | "
            f"Ch2: Otsu={T2o:.3g}, p={p2_used:.0f}%→{T2p:.3g} | "
            f"Range=[{args.pmin:.0f}–{args.pmax:.0f}]"
        )

    def save_viewer_screenshot(stem: str, out_dir: Path):
        if args.no_screenshot:
            return
        png_path = out_dir / f"{stem}_napari_overview.png"
        try:
            viewer.reset_view()
            loop = QEventLoop()
            QTimer.singleShot(int(args.screenshot_wait_ms), loop.quit)
            loop.exec_()
            img = viewer.screenshot(canvas_only=True, flash=False)

            # trim black borders
            if img.ndim == 3:
                gray = np.mean(img[:, :, :3], axis=2)
                mask = gray > 4.0
                if mask.any():
                    y_idx = np.where(mask.any(axis=1))[0]
                    x_idx = np.where(mask.any(axis=0))[0]
                    y0, y1 = y_idx[0], y_idx[-1]
                    x0, x1 = x_idx[0], x_idx[-1]
                    img = img[y0:y1+1, x0:x1+1, :]
            import imageio.v2 as iio
            iio.imwrite(png_path, img)
        except Exception as e:
            print(f"[screenshot] {stem}: failed → {e}")

    def process_file(fp: Path):
        img = load_image(str(fp))
        CZYX = to_CZYX(img)

        if ch1 >= CZYX.shape[0] or ch2 >= CZYX.shape[0]:
            raise ValueError(f"File {fp.name}: has {CZYX.shape[0]} channels; need {ch1},{ch2}")

        add_layers(CZYX, fp.stem)
        save_viewer_screenshot(fp.stem, out_dir)

        n_unmasked, n_masked = export_scatter_pairs_and_previews(
            CZYX, ch1, ch2, fp.stem, out_dir,
            do_hexbin=args.hexbin, preview_max=args.preview_max,
            draw_fit=(not args.no_preview_fit), ref_mode=args.preview_ref,
            png_masked_only=args.png_masked_only,
            mask_mode=args.mask_mode, p1=args.p1, p2=args.p2, pmin=args.pmin, pmax=args.pmax,
            red_inc_frac=args.red_inclusive_frac
        )

        status.value = (f"Wrote UNMASKED {n_unmasked:,} + ColocZ {n_masked:,} pairs "
                        f"→ CSV/PNGs in {out_dir.name}")

    def step():
        if idx["i"] >= len(files):
            status.value = f"Done. Exported {len(files)} file(s) → {out_dir}"
            if args.close_when_done:
                viewer.close()
            return

        fp = files[idx["i"]]
        try:
            status.value = f"[{idx['i']+1}/{len(files)}] Processing {fp.name} (CUDA={used_cuda})"
            process_file(fp)
        except Exception as e:
            status.value = f"[{idx['i']+1}/{len(files)}] ERROR: {e}"
        finally:
            idx["i"] += 1

        if not controls.stop_now.value:
            QTimer.singleShot(max(0, int(controls.pause_ms.value)), step)

    QTimer.singleShot(50, step)
    napari.run()

# ---------- Headless path ----------
def run_headless(args):
    used_cuda = try_cuda(args.cuda)
    in_dir = Path(args.folder) if args.folder else choose_folder_gui()
    if not in_dir:
        print("No folder selected. Exiting.")
        return
    out_dir = Path(args.out) if args.out else (in_dir / "scatter_exports")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(in_dir.glob(args.pattern)))
    if not files:
        print(f"No files matching {args.pattern} in {in_dir}")
        return

    print(f"CUDA active: {used_cuda}")
    print(f"Found {len(files)} files. Writing to: {out_dir}\n")
    for i, fp in enumerate(files, 1):
        try:
            img = load_image(str(fp))
            CZYX = to_CZYX(img)
            if args.ch1 >= CZYX.shape[0] or args.ch2 >= CZYX.shape[0]:
                print(f"[{i}/{len(files)}] {fp.name}: skip (has {CZYX.shape[0]} channels; need {args.ch1},{args.ch2})")
                continue

            n_unmasked, n_masked = export_scatter_pairs_and_previews(
                CZYX, args.ch1, args.ch2, fp.stem, out_dir,
                do_hexbin=args.hexbin, preview_max=args.preview_max,
                draw_fit=(not args.no_preview_fit), ref_mode=args.preview_ref,
                png_masked_only=args.png_masked_only,
                mask_mode=args.mask_mode, p1=args.p1, p2=args.p2, pmin=args.pmin, pmax=args.pmax,
                red_inc_frac=args.red_inclusive_frac
            )
            print(f"[{i}/{len(files)}] {fp.name}: UNMASKED {n_unmasked:,} | ColocZ {n_masked:,}")
        except Exception as e:
            print(f"[{i}/{len(files)}] {fp.name}: ERROR → {e}")
    print("\nDone.")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="Napari batch viewer + scatter CSV exporter (UNMASKED + ColocZ masked; CUDA optional)."
    )
    ap.add_argument("--folder", type=str, default="", help="Input folder (GUI if blank).")
    ap.add_argument("--out", type=str, default="", help="Output folder (default: <folder>/scatter_exports).")
    ap.add_argument("--pattern", type=str, default="*.tif", help="Glob pattern, e.g. *.tif or *.tiff")
    ap.add_argument("--ch1", type=int, default=0, help="Channel index for X (default 0)")
    ap.add_argument("--ch2", type=int, default=1, help="Channel index for Y (default 1)")
    ap.add_argument("--pause-ms", type=int, default=800, help="Pause between files (ms).")
    ap.add_argument("--cuda", action="store_true", help="Try to use CuPy (GPU) for small array ops.")
    ap.add_argument("--headless", action="store_true", help="Do not open Napari; just export.")
    ap.add_argument("--close-when-done", action="store_true", help="Close viewer automatically when done.")
    ap.add_argument("--hexbin", action="store_true", help="Hexbin previews instead of scatter.")
    ap.add_argument("--preview-max", type=int, default=0, help="Max points in preview PNGs (0 = all).")
    ap.add_argument("--no-preview-fit", action="store_true", help="Disable fitted preview line on the scatter PNGs.")
    ap.add_argument("--preview-ref", type=str, default="none",
                    help="Dashed reference on previews: 'none' | 'unit' | 'gain:<K>' (e.g., gain:2).")
    ap.add_argument("--no-screenshot", action="store_true", help="Disable per-file Napari screenshot.")
    ap.add_argument("--screenshot-wait-ms", type=int, default=300, help="Time to wait before screenshot (ms).")
    ap.add_argument("--png-masked-only", action="store_true",
                    help="Only write the ColocZ preview PNG; still export BOTH CSVs.")

    # Mask configuration (with windowed range)
    ap.add_argument("--mask-mode", type=str, default="hybrid",
                    choices=["otsu","percentile","hybrid"],
                    help="How to threshold each channel: 'otsu', 'percentile', or 'hybrid' (max of both).")
    ap.add_argument("--p1", type=float, default=50.0, help="Target percentile for Ch1 (clamped to [pmin,pmax]).")
    ap.add_argument("--p2", type=float, default=50.0, help="Target percentile for Ch2 (clamped to [pmin,pmax]).")
    ap.add_argument("--pmin", type=float, default=35.0, help="Lower bound for allowed percentile range.")
    ap.add_argument("--pmax", type=float, default=70.0, help="Upper bound for allowed percentile range.")

    # NEW: red-channel inclusivity factor (0..1). 0.40 = 40% more inclusive for Ch1.
    ap.add_argument("--red-inclusive-frac", type=float, default=0.40,
                    help="Fraction (0..1) to make Ch1 (red) more inclusive; "
                         "pulls Ch1 thresholds down toward the pmin baseline. Default 0.40.")

    args = ap.parse_args()

    # Clamp inclusivity
    if not (0.0 <= args.red_inclusive_frac <= 1.0):
        raise SystemExit("--red-inclusive-frac must be in [0, 1].")

    if args.headless:
        run_headless(args)
    else:
        run_with_viewer(args)

if __name__ == "__main__":
    main()
