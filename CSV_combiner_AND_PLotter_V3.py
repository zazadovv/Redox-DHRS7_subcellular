#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pool per-file scatter CSVs (Masked=ColocZ + Unmasked), compute pooled regression (+95% CI),
export per-file and summary stats (incl. Pearson & Manders), and write 3 figures:

  - pooled_regression_overlay.png            (Masked, raw scale; dashed 0-intercept line)
  - pooled_regression_overlay_minmax.png     (Masked, 0–1; dashed y=x)
  - unmasked_regression_overlay_minmax.png   (Unmasked, 0–1; dashed y=x)

Excel workbook: pooled_results.xlsx
  Sheets:
    - Masked_per_file
    - Masked_summary
    - Unmasked_per_file
    - Unmasked_summary
    - Coloc_calls

Notes
-----
* Equal weight per file for pooled line: average of per-file [intercept, slope].
* 95% CI band computed from **raw (unconstrained)** per-file coefficients across files.
* Center line uses **nonnegative** slope by default (slope>=0), but you can allow negatives.
* Manders for Masked needs Unmasked partner files in same folder (name swap).
* Minmax plots normalize scatter and the pooled line/CI to [0,1] for display only.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------- GUI folder picker ----------------
def pick_folder_gui() -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk(); root.withdraw()
    sel = filedialog.askdirectory(title="Select folder that contains the per-file scatter CSVs")
    return Path(sel) if sel else None


# ---------------- column detection ----------------
def detect_xy_columns(cols: List[str]) -> Tuple[Optional[str], Optional[str]]:
    low = {c.lower(): c for c in cols}
    candidates = [
        ("ch1","ch2"),("channel1","channel2"),
        ("x","y"),("intensity_ch1","intensity_ch2"),
        ("a","b"),
    ]
    for a,b in candidates:
        if a in low and b in low:
            return low[a], low[b]
    return None, None


# ---------------- OLS helpers ----------------
def ols_free_from_sums(n, sx, sy, sxx, sxy) -> Tuple[float,float]:
    den = n * sxx - sx * sx
    if den == 0 or n < 2:
        return np.nan, np.nan
    slope = (n * sxy - sx * sy) / den
    intercept = (sy - slope * sx) / n
    return float(slope), float(intercept)

def ols_zero_from_sums(sxx, sxy) -> float:
    if sxx <= 0:
        return np.nan
    return float(sxy / sxx)

def pearson_from_sums(n, sx, sy, sxx, syy, sxy) -> float:
    if n < 2: return np.nan
    num = n*sxy - sx*sy
    denx = n*sxx - sx*sx
    deny = n*syy - sy*sy
    den = np.sqrt(max(denx,0.0) * max(deny,0.0))
    if den == 0: return np.nan
    return float(num / den)


# ---------------- data scanning ----------------
def scan_file_for_ols_and_points(fp: Path,
    x_hint: Optional[str], y_hint: Optional[str],
    want_points: bool, max_points: int
) -> Tuple[float,float,float,float,int,np.ndarray,np.ndarray,float,float]:
    n = 0; sx = sy = sxx = sxy = syy = 0.0
    xs_list = []; ys_list = []; taken = 0
    rng = np.random.RandomState(0)
    x_min = np.inf; x_max = -np.inf

    for chunk in pd.read_csv(fp, chunksize=500_000, dtype=float, engine="c", low_memory=False):
        cols = list(chunk.columns)
        x_col = x_hint; y_col = y_hint
        if x_col is None or y_col is None:
            x_col, y_col = detect_xy_columns(cols)
            if not x_col or not y_col:
                num_cols = [c for c in cols if np.issubdtype(chunk[c].dtype, np.number)]
                if len(num_cols) < 2: continue
                x_col, y_col = num_cols[0], num_cols[1]

        x = chunk[x_col].to_numpy(dtype=np.float64, copy=False)
        y = chunk[y_col].to_numpy(dtype=np.float64, copy=False)
        good = np.isfinite(x) & np.isfinite(y)
        if not np.any(good): continue
        x = x[good]; y = y[good]

        n_chunk = x.size
        n   += n_chunk
        sx  += float(np.sum(x)); sy  += float(np.sum(y))
        sxx += float(np.sum(x*x)); sxy += float(np.sum(x*y)); syy += float(np.sum(y*y))

        x_min = min(x_min, float(np.min(x))); x_max = max(x_max, float(np.max(x)))

        if want_points:
            if max_points <= 0:
                xs_list.append(x.copy()); ys_list.append(y.copy())
            else:
                remaining = max_points - taken
                if remaining > 0:
                    if n_chunk <= remaining:
                        xs_list.append(x.copy()); ys_list.append(y.copy()); taken += n_chunk
                    else:
                        idx = rng.choice(n_chunk, size=remaining, replace=False)
                        xs_list.append(x[idx]); ys_list.append(y[idx]); taken = max_points

    slope_free, intercept_free = ols_free_from_sums(n, sx, sy, sxx, sxy)
    slope0 = ols_zero_from_sums(sxx, sxy)
    r = pearson_from_sums(n, sx, sy, sxx, syy, sxy)

    Xplot = np.concatenate(xs_list) if xs_list else np.empty((0,), dtype=np.float64)
    Yplot = np.concatenate(ys_list) if ys_list else np.empty((0,), dtype=np.float64)
    return slope_free, intercept_free, slope0, r, n, Xplot, Yplot, x_min, x_max


# ---------------- Manders ----------------
def manders_from_pairs(masked_fp: Path, unmasked_fp: Path,
    x_hint: Optional[str], y_hint: Optional[str]) -> Tuple[float,float]:
    if not unmasked_fp.exists(): return np.nan, np.nan
    def sums(fp: Path) -> Tuple[float,float]:
        s1=s2=0.0
        for chunk in pd.read_csv(fp, chunksize=500_000, dtype=float, engine="c", low_memory=False):
            cols = list(chunk.columns)
            x_col = x_hint; y_col = y_hint
            if x_col is None or y_col is None:
                x_col, y_col = detect_xy_columns(cols)
                if not x_col or not y_col:
                    num_cols = [c for c in cols if np.issubdtype(chunk[c].dtype, np.number)]
                    if len(num_cols) < 2: continue
                    x_col, y_col = num_cols[0], num_cols[1]
            x = chunk[x_col].to_numpy(dtype=np.float64, copy=False)
            y = chunk[y_col].to_numpy(dtype=np.float64, copy=False)
            good = np.isfinite(x) & np.isfinite(y)
            if not np.any(good): continue
            x = x[good]; y = y[good]
            s1 += float(np.sum(x)); s2 += float(np.sum(y))
        return s1, s2
    m1_num,m2_num = sums(masked_fp); m1_den,m2_den = sums(unmasked_fp)
    M1 = float(m1_num/m1_den) if m1_den>0 else np.nan
    M2 = float(m2_num/m2_den) if m2_den>0 else np.nan
    return M1,M2


# ---------------- plotting ----------------
def plot_overlay(X_all, Y_all, x_line, y_center, y_lo, y_hi,
    slope_label, out_png, title, dashed=None, fix_unit_square=False):

    fig, ax = plt.subplots(figsize=(8, 8))
    if X_all.size: ax.scatter(X_all, Y_all, s=1, alpha=0.10, linewidths=0,
                              color="#1f77b4", zorder=1)

    ax.fill_between(x_line, y_lo, y_hi, color="0.92", alpha=0.35, lw=0, label="95% CI (mean)", zorder=2)
    ax.plot(x_line, y_center, color="black", lw=2.4, label="pooled fit (free intercept)", zorder=3)
    if dashed is not None: ax.plot(dashed[0], dashed[1], color="black", lw=2.0, ls="--", label="guide", zorder=3)

    leg = ax.legend(loc="upper left", frameon=True)
    leg.get_frame().set_alpha(0.85); leg.get_frame().set_facecolor("white"); leg.get_frame().set_edgecolor("none")

    ax.text(0.98, 0.98, slope_label, transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, boxstyle="round,pad=0.25"),
            zorder=4)
    ax.set_title(title); ax.set_xlabel("Channel 1 intensity"); ax.set_ylabel("Channel 2 intensity")

    if fix_unit_square:
        ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.0); ax.set_aspect("equal", adjustable="box")

    ax.margins(x=0.02, y=0.05)
    fig.tight_layout(); fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)


# ---------------- pooled-line/CI helpers ----------------
def pooled_line_and_ci_from_coeffs(B_raw, x_min, x_max, grid):
    n_files = B_raw.shape[0]
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max<=x_min:
        x_min,x_max = 0.0,1.0
    xg = np.linspace(x_min, x_max, max(50,int(grid)))
    b_mean = np.nanmean(B_raw, axis=0)
    cov = np.cov(B_raw.T, ddof=1) if n_files>1 else np.zeros((2,2), float)
    cov_mean = cov / max(n_files,1)
    y_mean = b_mean[0]+b_mean[1]*xg
    var_y = cov_mean[0,0]+(xg**2)*cov_mean[1,1]+2.0*xg*cov_mean[0,1]
    se_y = np.sqrt(np.maximum(var_y,0.0))
    try:
        from scipy.stats import t
        tcrit = t.ppf(0.975, df=max(n_files-1,1))
    except Exception: tcrit=1.96
    y_lo = y_mean - tcrit*se_y; y_hi = y_mean + tcrit*se_y
    def mean_ci(a):
        m=float(np.nanmean(a))
        if n_files>1:
            sd=float(np.nanstd(a, ddof=1)); se=sd/np.sqrt(n_files)
            lo=m-tcrit*se; hi=m+tcrit*se
        else: lo=hi=np.nan
        return m,lo,hi
    slope_mean,slope_lo,slope_hi = mean_ci(B_raw[:,1])
    intercept_mean,_,_ = mean_ci(B_raw[:,0])
    return xg,y_mean,y_lo,y_hi,slope_mean,slope_lo,slope_hi,intercept_mean,tcrit


# ---------------- process family ----------------
def process_family(root,pattern,title_suffix,x_hint,y_hint,plot_max,grid,
    use_nonneg_center_slope,do_raw_plot,do_minmax_plot,out_prefix,
    pearson_thr,manders_thr,compute_manders):

    files = sorted(root.rglob(pattern))
    if not files: return {"per_file_df": None, "summary_df": None}
    coefs=[]; X_plot=[]; Y_plot=[]; global_min=np.inf; global_max=-np.inf
    per_file_quota = max(0,plot_max//len(files)) if plot_max>0 else 0

    for fp in files:
        slope_f,intercept_f,slope0,r,n,x_pts,y_pts,mn,mx = scan_file_for_ols_and_points(
            fp,x_hint,y_hint,True,per_file_quota)
        if np.isfinite(slope_f) and slope_f<0.0:
            slope_fc=0.0; intercept_fc=float(np.nanmean(y_pts)) if y_pts.size else intercept_f
        else: slope_fc,intercept_fc=slope_f,intercept_f
        if compute_manders:
            unm=fp.with_name(fp.name.replace("_scatter_points_ColocZ.csv","_scatter_points_unmasked.csv"))
            M1,M2 = manders_from_pairs(fp,unm,x_hint,y_hint)
            manders_call="Yes" if (np.isfinite(M1) and np.isfinite(M2) and M1>=manders_thr and M2>=manders_thr) else "No"
        else: M1=M2=np.nan; manders_call="N/A"
        pearson_call="Yes" if (np.isfinite(r) and r>=pearson_thr) else "No"
        if np.isfinite(slope_f) or np.isfinite(slope0):
            coefs.append((fp.name,int(n),float(slope_f),float(intercept_f),float(slope0),float(r),
                          float(slope_fc),float(intercept_fc),
                          f"y = {slope_f:.6g} * x + {intercept_f:.6g}",
                          f"y = {slope_fc:.6g} * x + {intercept_fc:.6g}",
                          float(M1),float(M2),pearson_call,manders_call))
        if x_pts.size: X_plot.append(x_pts); Y_plot.append(y_pts)
        if np.isfinite(mn): global_min=min(global_min,mn)
        if np.isfinite(mx): global_max=max(global_max,mx)

    if not coefs: return {"per_file_df": None,"summary_df": None}
    per_cols=["file","n","slope_free","intercept_free","slope0","pearson_r",
              "slope_free_constrained","intercept_free_constrained",
              "equation_free","equation_free_constrained","Manders_M1","Manders_M2",
              "Pearson_call","Manders_call"]
    per_df=pd.DataFrame(coefs,columns=per_cols)

    B_raw=per_df[["intercept_free","slope_free"]].to_numpy(float)
    xg,y_mean_raw,y_lo,y_hi,slope_mean_raw,slope_lo,slope_hi,intercept_mean_raw,tcrit= \
        pooled_line_and_ci_from_coeffs(B_raw,global_min,global_max,grid)

    if use_nonneg_center_slope:
        slope_center=float(np.nanmean(per_df["slope_free_constrained"].to_numpy(float)))
    else: slope_center=slope_mean_raw
    intercept_center=intercept_mean_raw
    y_center=intercept_center+slope_center*xg

    def mean_ci(arr):
        m=float(np.nanmean(arr)); n_files=arr.size
        if n_files>1:
            sd=float(np.nanstd(arr,ddof=1)); se=sd/np.sqrt(n_files)
            lo=m-tcrit*se; hi=m+tcrit*se
        else: lo=hi=np.nan
        return m,lo,hi
    slope_used=(per_df["slope_free_constrained"] if use_nonneg_center_slope else per_df["slope_free"]).to_numpy(float)
    intercept_used=(per_df["intercept_free_constrained"] if use_nonneg_center_slope else per_df["intercept_free"]).to_numpy(float)
    slope_mean_rep,slope_lo_rep,slope_hi_rep=mean_ci(slope_used)
    intercept_mean_rep,intercept_lo_rep,intercept_hi_rep=mean_ci(intercept_used)
    slope0_mean,slope0_lo,slope0_hi=mean_ci(per_df["slope0"].to_numpy(float))

    summary_df=pd.DataFrame([{
        "n_files": int(per_df.shape[0]),
        "slope_free_mean_reported": slope_mean_rep,
        "slope_free_lo95_reported": slope_lo_rep,
        "slope_free_hi95_reported": slope_hi_rep,
        "intercept_free_mean_reported": intercept_mean_rep,
        "intercept_free_lo95_reported": intercept_lo_rep,
        "intercept_free_hi95_reported": intercept_hi_rep,
        "slope0_mean": slope0_mean,"slope0_lo95": slope0_lo,"slope0_hi95": slope0_hi,
        "slope_free_mean_raw_for_CI": slope_mean_raw,
        "slope_free_lo95_raw": slope_lo,"slope_free_hi95_raw": slope_hi
    }])

    # Save pooled line CSV (raw-coeff mean + CI)
    pd.DataFrame({"x": xg,"y_mean_free": y_mean_raw,"y_lo95_free": y_lo,"y_hi95_free": y_hi}).to_csv(
        root/f"pooled_regression_line_{out_prefix}.csv",index=False)

    # Build plots
    Xp=np.concatenate(X_plot) if X_plot else np.empty((0,),float)
    Yp=np.concatenate(Y_plot) if Y_plot else np.empty((0,),float)
    slope_label=f"slope (mean, reported) = {slope_mean_rep:.6g}  [95% CI from raw: {slope_lo:.6g}–{slope_hi:.6g}]"

    # (A) Raw-scale (masked only)
    if do_raw_plot:
        xd=xg
        yd=(slope0_mean*xd) if np.isfinite(slope0_mean) else np.full_like(xd,np.nan)
        title_raw=f"Pooled linear regression {title_suffix} (equal weight per file) + 95% CI"
        plot_overlay(Xp,Yp,xg,y_center,y_lo,y_hi,slope_label,
                     root/"pooled_regression_overlay.png", title=title_raw, dashed=(xd,yd), fix_unit_square=False)

    # (B) Minmax 0–1 with dashed y=x
    if do_minmax_plot and Xp.size:
        xmin=float(np.nanmin(Xp)); xmax=float(np.nanmax(Xp))
        ymin=float(np.nanmin(Yp)); ymax=float(np.nanmax(Yp))
        xr=(xmax-xmin) if xmax>xmin else 1.0
        yr=(ymax-ymin) if ymax>ymin else 1.0
        Xn=(Xp-xmin)/xr; Yn=(Yp-ymin)/yr
        xgn=np.linspace(0.0,1.0,max(50,int(grid)))
        y_center_n=(y_center-ymin)/yr; y_lo_n=(y_lo-ymin)/yr; y_hi_n=(y_hi-ymin)/yr
        title_mm=f"Pooled linear regression (MINMAX 0–1) {title_suffix} + 95% CI"
        out_png = root/(f"{out_prefix}_regression_overlay_minmax.png")
        plot_overlay(Xn,Yn,xgn,y_center_n,y_lo_n,y_hi_n,slope_label,out_png,
                     title=title_mm, dashed=(xgn,xgn), fix_unit_square=True)

    return {
        "per_file_df": per_df, "summary_df": summary_df,
        "Xplot": Xp, "Yplot": Yp,
        "x_min": global_min, "x_max": global_max,
        "xg": xg, "y_center": y_center, "y_lo": y_lo, "y_hi": y_hi,
        "slope_mean_raw": slope_mean_raw, "slope_lo": slope_lo, "slope_hi": slope_hi,
        "intercept_mean_raw": intercept_mean_raw
    }


# ---------------- main ----------------
def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Pooled regression (masked & unmasked) + 95% CI, per-file exports, Excel, and 3 figures."
    )
    ap.add_argument("--root", type=str, default="", help="Folder with per-file scatter CSVs (recursive). If blank, GUI picker opens.")
    ap.add_argument("--pattern_masked", type=str, default="*_scatter_points_ColocZ.csv",
                    help="Glob pattern for masked (ColocZ) CSVs.")
    ap.add_argument("--pattern_unmasked", type=str, default="*_scatter_points_unmasked.csv",
                    help="Glob pattern for unmasked CSVs.")
    ap.add_argument("--x-col", type=str, default="", help="Force X column name (optional).")
    ap.add_argument("--y-col", type=str, default="", help="Force Y column name (optional).")
    ap.add_argument("--plot-max", type=int, default=0, help="Max points TOTAL to plot across ALL files (0 = ALL).")
    ap.add_argument("--grid", type=int, default=200, help="Number of x points for drawing the line/CI.")
    ap.add_argument("--allow-neg-slope", dest="nonneg_slope", action="store_false",
                    help="Allow pooled center slope to go negative (by default it's constrained to >=0).")
    ap.set_defaults(nonneg_slope=True)
    ap.add_argument("--pearson-thr", type=float, default=0.10, help="Pearson threshold for 'colocalized?'.")
    ap.add_argument("--manders-thr", type=float, default=0.20, help="Manders threshold (applied to BOTH M1 and M2).")
    ap.add_argument("--no-minmax", action="store_true", help="Do not write MINMAX plots (masked/unmasked).")
    args = ap.parse_args()

    root = Path(args.root) if args.root else pick_folder_gui()
    if not root:
        print("No folder selected."); return

    x_hint = args.x_col if args.x_col else None
    y_hint = args.y_col if args.y_col else None

    # ---- MASKED family (ColocZ)
    masked = process_family(
        root=root,
        pattern=args.pattern_masked,
        title_suffix="(ColocZ)",
        x_hint=x_hint,
        y_hint=y_hint,
        plot_max=args.plot_max,
        grid=args.grid,
        use_nonneg_center_slope=bool(args.nonneg_slope),
        do_raw_plot=True,
        do_minmax_plot=(not args.no_minmax),
        out_prefix="pooled",
        pearson_thr=float(args.pearson_thr),
        manders_thr=float(args.manders_thr),
        compute_manders=True,
    )

    # ---- UNMASKED family
    unmasked = process_family(
        root=root,
        pattern=args.pattern_unmasked,
        title_suffix="(UNMASKED)",
        x_hint=x_hint,
        y_hint=y_hint,
        plot_max=args.plot_max,
        grid=args.grid,
        use_nonneg_center_slope=bool(args.nonneg_slope),
        do_raw_plot=False,                       # only minmax plot for unmasked
        do_minmax_plot=(not args.no_minmax),
        out_prefix="unmasked",
        pearson_thr=float(args.pearson_thr),
        manders_thr=float(args.manders_thr),
        compute_manders=False,                   # Manders computed only for masked vs unmasked
    )

    # ---- Excel export (5 sheets)
    out_xlsx = root / "pooled_results.xlsx"
    engine = None
    for cand in ("openpyxl", "xlsxwriter"):
        try:
            __import__(cand); engine = cand; break
        except Exception:
            continue
    if engine is None: engine = "xlsxwriter"

    with pd.ExcelWriter(out_xlsx, engine=engine) as writer:
        if masked["per_file_df"] is not None:
            masked["per_file_df"].to_excel(writer, sheet_name="Masked_per_file", index=False)
            masked["summary_df"].to_excel(writer, sheet_name="Masked_summary", index=False)
        if unmasked["per_file_df"] is not None:
            unmasked["per_file_df"].to_excel(writer, sheet_name="Unmasked_per_file", index=False)
            unmasked["summary_df"].to_excel(writer, sheet_name="Unmasked_summary", index=False)

        # Coloc_calls sheet from masked metrics
        if masked["per_file_df"] is not None:
            calls = masked["per_file_df"][["file","pearson_r","Manders_M1","Manders_M2","Pearson_call","Manders_call"]].copy()
            calls.rename(columns={"pearson_r":"pearson_r_masked"}, inplace=True)
            calls["Both_agree"] = np.where(
                (calls["Pearson_call"] == "Yes") & (calls["Manders_call"] == "Yes"), "Yes", "No"
            )
            calls["Final_call"] = np.where(calls["Both_agree"] == "Yes", "Yes", "No")
            calls.insert(1, "pearson_thr", float(args.pearson_thr))
            calls.insert(2, "manders_thr", float(args.manders_thr))
            calls.to_excel(writer, sheet_name="Coloc_calls", index=False)

    print(
        "Saved:\n"
        f"  {root/'pooled_regression_overlay.png'}\n"
        f"  {root/'pooled_regression_overlay_minmax.png'}\n"
        f"  {root/'unmasked_regression_overlay_minmax.png'}\n"
        f"  {root/'pooled_regression_line_pooled.csv'}   (masked line for raw/CI)\n"
        f"  {root/'pooled_regression_line_unmasked.csv'} (unmasked line for raw/CI)\n"
        f"  {out_xlsx}\n"
    )


if __name__ == "__main__":
    main()
