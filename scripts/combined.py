#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ============================================================
# FINAL 1-RQ EIS FITTER WITH WEIGHTING COMPARISON
# ------------------------------------------------------------
# Customized for uploaded files:
#   AS_EIS_for_fitting.csv
#   SR300_EIS_for_fitting.csv
#   SR400_EIS_for_fitting.csv
#   SR550_EIS_for_fitting.csv
#   SA1100_EIS_for_fitting.csv
#
# Physical model:
#   Rs - (Q || R)
#
# Optimization:
#   1) differential_evolution (global)
#   2) least_squares (local refinement)
#
# Weighting schemes compared:
#   - modulus
#   - separate
#   - none
#
# Accepted input column styles:
#   1) freq_Hz, ReZ_ohm_cm2, neg_ImZ_ohm_cm2
#   2) Freq[Hz], ReZx[Ohm*cm2], -ImZx[Ohm*cm2]
#
# Output:
#   - one subfolder per weighting scheme
#   - per-sample plots and fit CSVs
#   - per-weighting summary CSV
#   - global comparison table of all weightings
#   - combined Nyquist + Bode figure per sample
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import least_squares, differential_evolution

# -----------------------------
# USER SETTINGS
# -----------------------------
INPUT_DIR = Path.cwd()
FALLBACK_INPUT_DIR = Path("/mnt/data") if Path("/mnt/data").exists() else None

FILE_CANDIDATES = {
    "AS": [
        "AS_EIS_for_fitting.csv",
    ],
    "SR300": [
        "SR300_EIS_for_fitting.csv",
    ],
    "SR400": [
        "SR400_EIS_for_fitting.csv",
    ],
    "SR550": [
        "SR550_EIS_for_fitting.csv",
    ],
    "SA1100": [
        "SA1100_EIS_for_fitting.csv",
    ],
}

PLOT_ORDER = ["AS", "SR300", "SR400", "SR550", "SA1100"]

# Final restrained trimming for the uploaded dataset
FIT_OVERRIDES = {
    "AS":     {"trim_high": 0, "trim_low": 0, "use_robust_loss": False},
    "SR300":  {"trim_high": 0, "trim_low": 1, "use_robust_loss": True},
    "SR400":  {"trim_high": 2, "trim_low": 0, "use_robust_loss": True},
    "SR550":  {"trim_high": 0, "trim_low": 2, "use_robust_loss": True},
    "SA1100": {"trim_high": 0, "trim_low": 0, "use_robust_loss": False},
}

WEIGHTING_SCHEMES = ["modulus", "separate", "none"]

OUT_DIR = INPUT_DIR / "EIS_1RQ_weighting_comparison_uploaded_files"

SAVE_PLOTS = True
SAVE_FIT_CSV = True
SAVE_RESIDUAL_PLOTS = True
SAVE_COMBINED_PLOTS = True

FIG_DPI = 1200
DENSE_FREQ_POINTS = 5000

# -----------------------------
# GLOBAL FITTING OPTIONS
# -----------------------------
GLOBAL_USE_ROBUST_LOSS = False
ROBUST_LOSS = "soft_l1"
ROBUST_F_SCALE = 1.0

# Physically reasonable bounds
N_BOUNDS = (0.70, 1.00)
RS_UPPER_FACTOR = 20.0
R_UPPER_FACTOR = 50.0
Q_LOWER = 1e-12
Q_UPPER = 1e-1

# Hybrid optimizer settings
DE_MAXITER = 300
DE_POPSIZE = 20
DE_TOL = 1e-7
DE_POLISH = False
DE_SEED = 42

# -----------------------------
# PLOT STYLE
# -----------------------------
EXP_COLOR = "#0000FF"
FIT_COLOR = "#FF0000"

EXP_MARKER_SIZE = 8.5
FIT_LINEWIDTH = 2.8

GRID_COLOR = "#b0b0b0"
GRID_ALPHA = 0.6
GRID_WIDTH = 1.0
SPINE_COLOR = "#000000"

BODE_XMIN_HZ = 1e-2
BODE_MAG_YMIN_OHM = 1e-2
BODE_MAG_YMAX_OHM = 1e6

PHASE_MAJOR_STEP = 20.0
PHASE_MINOR_STEP = 10.0

plt.rcParams.update({
    "figure.dpi": 220,
    "savefig.dpi": FIG_DPI,
    "font.size": 13,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.linewidth": 2.0,
    "xtick.major.width": 1.8,
    "ytick.major.width": 1.8,
    "xtick.minor.width": 1.2,
    "ytick.minor.width": 1.2,
    "xtick.major.size": 8,
    "ytick.major.size": 8,
    "xtick.minor.size": 4,
    "ytick.minor.size": 4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "mathtext.default": "regular",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "path.simplify": False,
    "lines.antialiased": True,
    "patch.antialiased": True,
    "text.antialiased": True,
})

# -----------------------------
# HELPERS
# -----------------------------
def apply_bold_ticks(ax):
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")


def apply_bold_legend(ax):
    leg = ax.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontweight("bold")


def apply_reference_axes_style(ax, use_minor=True):
    ax.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=GRID_WIDTH)
    if use_minor:
        ax.minorticks_on()
    else:
        ax.minorticks_off()

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_edgecolor(SPINE_COLOR)


def style_axis(ax, logx=False, logy=False, format_log_x=False, format_log_y=False, use_minor=True):
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    if format_log_x:
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0))
        ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
        ax.xaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
        )

    if format_log_y:
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
        )

    apply_reference_axes_style(ax, use_minor=use_minor)


def save_figure_png(fig, out_base: Path):
    fig.savefig(f"{out_base}.png", dpi=FIG_DPI, bbox_inches="tight")


def round_up_axis_limit(max_value):
    if max_value <= 1:
        step = 0.2
    elif max_value <= 2:
        step = 0.5
    elif max_value <= 5:
        step = 1
    elif max_value <= 10:
        step = 2
    elif max_value <= 25:
        step = 5
    elif max_value <= 50:
        step = 10
    elif max_value <= 100:
        step = 20
    elif max_value <= 250:
        step = 50
    elif max_value <= 500:
        step = 100
    else:
        step = 200
    upper = np.ceil(max_value / step) * step
    return max(upper, step), step


def get_fit_options(sample_name: str):
    override = FIT_OVERRIDES.get(sample_name, {})
    trim_high = int(override.get("trim_high", 0))
    trim_low = int(override.get("trim_low", 0))
    use_robust_loss = bool(override.get("use_robust_loss", GLOBAL_USE_ROBUST_LOSS))
    return trim_high, trim_low, use_robust_loss


def resolve_input_file(candidates):
    for filename in candidates:
        f1 = INPUT_DIR / filename
        if f1.exists():
            return f1
        if FALLBACK_INPUT_DIR is not None:
            f2 = FALLBACK_INPUT_DIR / filename
            if f2.exists():
                return f2
    return None


# -----------------------------
# DATA READER
# -----------------------------
def read_eis_csv(csv_file: Path, trim_high: int, trim_low: int):
    df = pd.read_csv(csv_file)
    df.columns = [str(c).strip() for c in df.columns]

    freq_candidates = ["freq_Hz", "Freq[Hz]"]
    re_candidates = ["ReZ_ohm_cm2", "ReZx[Ohm*cm2]"]
    neg_im_candidates = ["neg_ImZ_ohm_cm2", "-ImZx[Ohm*cm2]"]
    im_candidates = ["ImZ_ohm_cm2"]

    def pick_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    freq_col = pick_col(freq_candidates)
    re_col = pick_col(re_candidates)
    neg_im_col = pick_col(neg_im_candidates)
    im_col = pick_col(im_candidates)

    if freq_col is None:
        raise KeyError(f"{csv_file.name}: frequency column not found.")
    if re_col is None:
        raise KeyError(f"{csv_file.name}: real impedance column not found.")

    if neg_im_col is not None:
        neg_im = pd.to_numeric(df[neg_im_col], errors="coerce")
    elif im_col is not None:
        neg_im = -pd.to_numeric(df[im_col], errors="coerce")
    else:
        raise KeyError(
            f"{csv_file.name}: neither neg_ImZ_ohm_cm2 / -ImZx[Ohm*cm2] nor ImZ_ohm_cm2 was found."
        )

    freq = pd.to_numeric(df[freq_col], errors="coerce")
    z_re = pd.to_numeric(df[re_col], errors="coerce")

    clean = pd.DataFrame({
        "freq_Hz": freq,
        "ReZ_ohm_cm2": z_re,
        "neg_ImZ_ohm_cm2": neg_im,
    }).dropna()

    clean = clean[clean["freq_Hz"] > 0].copy()
    clean = clean.sort_values("freq_Hz", ascending=False).reset_index(drop=True)

    n_before = len(clean)

    start = int(trim_high)
    end = len(clean) - int(trim_low)

    if start >= end:
        raise ValueError(
            f"Invalid trimming for {csv_file.name}: trim_high={trim_high}, "
            f"trim_low={trim_low}, n_points={len(clean)}"
        )

    df_fit = clean.iloc[start:end].reset_index(drop=True)
    n_after = len(df_fit)

    freq = df_fit["freq_Hz"].to_numpy(dtype=float)
    z_re = df_fit["ReZ_ohm_cm2"].to_numpy(dtype=float)
    neg_im = df_fit["neg_ImZ_ohm_cm2"].to_numpy(dtype=float)

    z_exp = z_re - 1j * neg_im
    return clean, df_fit, freq, z_exp, n_before, n_after


# -----------------------------
# MODEL
# -----------------------------
def z_model_rq(freq, Rs, R, Q, n):
    w = 2.0 * np.pi * np.asarray(freq, dtype=float)
    z_par = 1.0 / (1.0 / R + Q * (1j * w) ** n)
    return Rs + z_par


def residual_vector_rq(params, freq, z_exp, weighting="modulus"):
    Rs, R, Q, n = params
    z_fit = z_model_rq(freq, Rs, R, Q, n)

    err_re = z_fit.real - z_exp.real
    err_im = z_fit.imag - z_exp.imag

    if weighting == "modulus":
        w = np.maximum(np.abs(z_exp), 1e-15)
        res_re = err_re / w
        res_im = err_im / w

    elif weighting == "separate":
        w_re = max(np.max(np.abs(z_exp.real)), 1e-15)
        w_im = max(np.max(np.abs(z_exp.imag)), 1e-15)
        res_re = err_re / w_re
        res_im = err_im / w_im

    elif weighting == "none":
        res_re = err_re
        res_im = err_im

    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    return np.hstack([res_re, res_im])


def scalar_objective_rq(params, freq, z_exp, weighting="modulus"):
    r = residual_vector_rq(params, freq, z_exp, weighting=weighting)
    return np.sum(r ** 2)


# -----------------------------
# METRICS
# -----------------------------
def fit_metrics(z_exp, z_fit):
    err_re = z_fit.real - z_exp.real
    err_im = z_fit.imag - z_exp.imag

    rmse_re = np.sqrt(np.mean(err_re ** 2))
    rmse_im = np.sqrt(np.mean(err_im ** 2))

    wrmse_modulus = np.sqrt(
        np.mean((np.abs(z_fit - z_exp) / np.maximum(np.abs(z_exp), 1e-15)) ** 2)
    )

    return rmse_re, rmse_im, wrmse_modulus


def cpe_to_ceff_hsu_mansfeld(R, Q, n):
    return (Q * (R ** (1.0 - n))) ** (1.0 / n)


def bode_mag_phase(z):
    mag = np.abs(z)
    phase = -np.degrees(np.angle(z))
    return mag, phase


def make_dense_frequency(freq, n_points=DENSE_FREQ_POINTS):
    fmin = np.min(freq)
    fmax = np.max(freq)
    return np.logspace(np.log10(fmin), np.log10(fmax), n_points)


# -----------------------------
# HYBRID FITTER
# -----------------------------
def build_bounds(freq, z_exp):
    max_re = max(float(np.max(z_exp.real)), 1.0)
    rspan = max(float(np.max(z_exp.real) - np.min(z_exp.real)), 1e-6)

    lb = np.array([1e-9, 1e-6, Q_LOWER, N_BOUNDS[0]], dtype=float)
    ub = np.array([
        max_re * RS_UPPER_FACTOR,
        max(rspan * R_UPPER_FACTOR, 1e4),
        Q_UPPER,
        N_BOUNDS[1]
    ], dtype=float)

    return lb, ub


def fit_one_file_hybrid(freq, z_exp, weighting="modulus", use_robust_loss=False):
    lb, ub = build_bounds(freq, z_exp)
    bounds_de = list(zip(lb, ub))

    de_result = differential_evolution(
        func=scalar_objective_rq,
        bounds=bounds_de,
        args=(freq, z_exp, weighting),
        strategy="best1bin",
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        tol=DE_TOL,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=DE_SEED,
        polish=DE_POLISH,
        updating="deferred",
        workers=1,
    )

    x0 = np.clip(de_result.x, lb * 1.000001, ub * 0.999999)

    ls_kwargs = dict(
        fun=residual_vector_rq,
        x0=x0,
        args=(freq, z_exp, weighting),
        bounds=(lb, ub),
        method="trf",
        x_scale="jac",
        max_nfev=50000,
    )

    if use_robust_loss:
        ls_kwargs["loss"] = ROBUST_LOSS
        ls_kwargs["f_scale"] = ROBUST_F_SCALE
    else:
        ls_kwargs["loss"] = "linear"

    fit_res = least_squares(**ls_kwargs)

    n_data = 2 * len(freq)
    n_par = len(fit_res.x)

    try:
        jac = fit_res.jac
        dof = max(n_data - n_par, 1)
        s_sq = 2.0 * fit_res.cost / dof
        cov = s_sq * np.linalg.pinv(jac.T @ jac)
        diag_cov = np.diag(cov)
        diag_cov = np.where(diag_cov >= 0, diag_cov, np.nan)
        stderr = np.sqrt(diag_cov)
    except Exception:
        stderr = np.full(n_par, np.nan)

    return fit_res, stderr, de_result


# -----------------------------
# OUTPUT FILES
# -----------------------------
def save_fit_csv(out_dir, sample_name, weighting, freq, z_exp, z_fit):
    mag_exp, phase_exp = bode_mag_phase(z_exp)
    mag_fit, phase_fit = bode_mag_phase(z_fit)

    out = pd.DataFrame({
        "sample": sample_name,
        "weighting": weighting,
        "Freq_Hz": freq,
        "Zre_exp_ohm_cm2": z_exp.real,
        "neg_Zimag_exp_ohm_cm2": -z_exp.imag,
        "Zre_fit_ohm_cm2": z_fit.real,
        "neg_Zimag_fit_ohm_cm2": -z_fit.imag,
        "Residual_Zre_ohm_cm2": z_fit.real - z_exp.real,
        "Residual_neg_Zimag_ohm_cm2": (-z_fit.imag) - (-z_exp.imag),
        "|Z|_exp_ohm_cm2": mag_exp,
        "|Z|_fit_ohm_cm2": mag_fit,
        "Phase_exp_deg": phase_exp,
        "Phase_fit_deg": phase_fit,
    })
    out.to_csv(out_dir / f"{sample_name}_fit_data.csv", index=False)


# -----------------------------
# PLOTTING
# -----------------------------
def plot_nyquist(out_dir, sample_name, weighting, z_exp, z_fit_dense):
    fig, ax = plt.subplots(figsize=(8.0, 8.0))

    OHM_TO_KOHM = 1.0 / 1000.0

    x_exp = z_exp.real * OHM_TO_KOHM
    y_exp = (-z_exp.imag) * OHM_TO_KOHM
    x_fit = z_fit_dense.real * OHM_TO_KOHM
    y_fit = (-z_fit_dense.imag) * OHM_TO_KOHM

    ax.plot(
        x_exp, y_exp,
        linestyle="None",
        marker="o",
        markersize=EXP_MARKER_SIZE,
        markerfacecolor=EXP_COLOR,
        markeredgecolor=EXP_COLOR,
        label="Experimental",
        zorder=3,
    )

    ax.plot(
        x_fit, y_fit,
        linestyle="-",
        linewidth=FIT_LINEWIDTH,
        color=FIT_COLOR,
        label="ECM Fit",
        antialiased=True,
        solid_joinstyle="round",
        solid_capstyle="round",
        zorder=2,
    )

    ax.set_xlabel(
        r"$\vec{\mathbf{Z}}^\prime\;(\mathbf{k\Omega\cdot cm^{2}})$",
        fontweight="bold",
        fontsize=18,
    )
    ax.set_ylabel(
        r"$-\vec{\mathbf{Z}}^{\prime\prime}\;(\mathbf{k\Omega\cdot cm^{2}})$",
        fontweight="bold",
        fontsize=18,
    )
    ax.set_title(
        f"Nyquist Plot — {sample_name} ({weighting})",
        fontweight="bold",
        fontsize=22,
        pad=12
    )

    apply_reference_axes_style(ax, use_minor=False)

    x_max_data = max(np.nanmax(x_exp), np.nanmax(x_fit), 0.0)
    y_max_data = max(
        np.nanmax(np.clip(y_exp, 0, None)),
        np.nanmax(np.clip(y_fit, 0, None)),
        0.0
    )
    shared_upper, major_step = round_up_axis_limit(max(x_max_data, y_max_data))

    ax.set_xlim(0, shared_upper)
    ax.set_ylim(0, shared_upper)

    fmt = "{x:.1f}" if major_step < 1 else "{x:.0f}"
    ax.xaxis.set_major_locator(mticker.MultipleLocator(major_step))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(major_step))
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter(fmt))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(fmt))

    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", which="major", labelsize=16, width=1.6, length=0)

    ax.legend(loc="best", fontsize=18, frameon=False, handlelength=2.2)

    fig.subplots_adjust(left=0.16, right=0.96, top=0.90, bottom=0.14)

    fig.canvas.draw()
    apply_bold_ticks(ax)
    apply_bold_legend(ax)

    save_figure_png(fig, out_dir / f"{sample_name}_Nyquist_fit")
    plt.close(fig)


def plot_bode(out_dir, sample_name, weighting, freq, z_exp, freq_dense, z_fit_dense):
    mag_exp, phase_exp = bode_mag_phase(z_exp)
    mag_fit, phase_fit = bode_mag_phase(z_fit_dense)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 9.0), sharex=True)

    axes[0].plot(
        freq, mag_exp,
        linestyle="None",
        marker="o",
        markersize=EXP_MARKER_SIZE,
        markerfacecolor=EXP_COLOR,
        markeredgecolor=EXP_COLOR,
        label="Experimental",
        zorder=3,
    )
    axes[0].plot(
        freq_dense, mag_fit,
        linestyle="-",
        linewidth=FIT_LINEWIDTH,
        color=FIT_COLOR,
        label="ECM Fit",
        antialiased=True,
        solid_joinstyle="round",
        solid_capstyle="round",
        zorder=2,
    )

    axes[0].set_ylabel(
        r"$|\mathbf{Z}|\;(\mathbf{\Omega\cdot cm^{2}})$",
        fontweight="bold",
        fontsize=18,
        labelpad=12
    )
    axes[0].set_title(
        f"Bode Plot — {sample_name} ({weighting})",
        fontweight="bold",
        fontsize=22,
        pad=10
    )

    style_axis(
        axes[0],
        logx=True,
        logy=True,
        format_log_x=True,
        format_log_y=True,
        use_minor=True
    )

    axes[0].set_xlim(BODE_XMIN_HZ, np.max(freq_dense))
    axes[0].set_ylim(BODE_MAG_YMIN_OHM, BODE_MAG_YMAX_OHM)
    axes[0].tick_params(axis="x", which="both", labelbottom=True)
    axes[0].legend(loc="best", fontsize=18, frameon=False, handlelength=2.2)

    axes[1].plot(
        freq, phase_exp,
        linestyle="None",
        marker="o",
        markersize=EXP_MARKER_SIZE,
        markerfacecolor=EXP_COLOR,
        markeredgecolor=EXP_COLOR,
        label="Experimental",
        zorder=3,
    )
    axes[1].plot(
        freq_dense, phase_fit,
        linestyle="-",
        linewidth=FIT_LINEWIDTH,
        color=FIT_COLOR,
        label="ECM Fit",
        antialiased=True,
        solid_joinstyle="round",
        solid_capstyle="round",
        zorder=2,
    )

    axes[1].set_xlabel(
        r"$f\;(\mathrm{Hz})$",
        fontweight="bold",
        fontsize=18,
        labelpad=10
    )
    axes[1].set_ylabel(
        r"$-\phi\;(^\circ)$",
        fontweight="bold",
        fontsize=18,
        labelpad=12
    )

    style_axis(
        axes[1],
        logx=True,
        logy=False,
        format_log_x=True,
        format_log_y=False,
        use_minor=True
    )

    phase_min_data = min(np.nanmin(phase_exp), np.nanmin(phase_fit))
    phase_max_data = max(np.nanmax(phase_exp), np.nanmax(phase_fit))

    phase_lower = PHASE_MAJOR_STEP * np.floor((phase_min_data - 5.0) / PHASE_MAJOR_STEP)
    phase_upper = PHASE_MAJOR_STEP * np.ceil((phase_max_data + 5.0) / PHASE_MAJOR_STEP)

    axes[1].set_xlim(BODE_XMIN_HZ, np.max(freq_dense))
    axes[1].set_ylim(phase_lower, phase_upper)
    axes[1].yaxis.set_major_locator(mticker.MultipleLocator(PHASE_MAJOR_STEP))
    axes[1].yaxis.set_minor_locator(mticker.MultipleLocator(PHASE_MINOR_STEP))

    for ax in axes:
        ax.tick_params(axis="x", which="major", labelsize=16, width=1.6, pad=10)
        ax.tick_params(axis="y", which="major", labelsize=16, width=1.6, pad=10)
        ax.tick_params(axis="x", which="minor", pad=8)
        ax.tick_params(axis="y", which="minor", pad=8)

    fig.subplots_adjust(
        left=0.18,
        right=0.96,
        top=0.93,
        bottom=0.12,
        hspace=0.18
    )

    fig.canvas.draw()
    for ax in axes:
        apply_bold_ticks(ax)
        apply_bold_legend(ax)

    save_figure_png(fig, out_dir / f"{sample_name}_Bode_fit")
    plt.close(fig)


def plot_combined_eis_figure(out_dir, sample_name, weighting, freq, z_exp, freq_dense, z_fit_dense):
    mag_exp, phase_exp = bode_mag_phase(z_exp)
    mag_fit, phase_fit = bode_mag_phase(z_fit_dense)

    fig = plt.figure(figsize=(14.0, 8.6))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.05, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.28,
        hspace=0.20
    )

    ax_nyq = fig.add_subplot(gs[:, 0])
    ax_mag = fig.add_subplot(gs[0, 1])
    ax_phs = fig.add_subplot(gs[1, 1])

    OHM_TO_KOHM = 1.0 / 1000.0
    x_exp = z_exp.real * OHM_TO_KOHM
    y_exp = (-z_exp.imag) * OHM_TO_KOHM
    x_fit = z_fit_dense.real * OHM_TO_KOHM
    y_fit = (-z_fit_dense.imag) * OHM_TO_KOHM

    ax_nyq.plot(
        x_exp, y_exp,
        linestyle="None",
        marker="o",
        markersize=EXP_MARKER_SIZE,
        markerfacecolor=EXP_COLOR,
        markeredgecolor=EXP_COLOR,
        label="Experimental",
        zorder=3,
    )
    ax_nyq.plot(
        x_fit, y_fit,
        linestyle="-",
        linewidth=FIT_LINEWIDTH,
        color=FIT_COLOR,
        label="ECM Fit",
        antialiased=True,
        solid_joinstyle="round",
        solid_capstyle="round",
        zorder=2,
    )

    ax_nyq.set_xlabel(
        r"$\vec{\mathbf{Z}}^\prime\;(\mathbf{k\Omega\cdot cm^{2}})$",
        fontweight="bold",
        fontsize=17,
    )
    ax_nyq.set_ylabel(
        r"$-\vec{\mathbf{Z}}^{\prime\prime}\;(\mathbf{k\Omega\cdot cm^{2}})$",
        fontweight="bold",
        fontsize=17,
    )
    ax_nyq.set_title(
        f"{sample_name} — Nyquist + Bode ({weighting})",
        fontweight="bold",
        fontsize=20,
        pad=10
    )

    apply_reference_axes_style(ax_nyq, use_minor=False)

    x_max_data = max(np.nanmax(x_exp), np.nanmax(x_fit), 0.0)
    y_max_data = max(
        np.nanmax(np.clip(y_exp, 0, None)),
        np.nanmax(np.clip(y_fit, 0, None)),
        0.0
    )
    shared_upper, major_step = round_up_axis_limit(max(x_max_data, y_max_data))

    ax_nyq.set_xlim(0, shared_upper)
    ax_nyq.set_ylim(0, shared_upper)

    fmt = "{x:.1f}" if major_step < 1 else "{x:.0f}"
    ax_nyq.xaxis.set_major_locator(mticker.MultipleLocator(major_step))
    ax_nyq.yaxis.set_major_locator(mticker.MultipleLocator(major_step))
    ax_nyq.xaxis.set_major_formatter(mticker.StrMethodFormatter(fmt))
    ax_nyq.yaxis.set_major_formatter(mticker.StrMethodFormatter(fmt))

    ax_nyq.set_aspect("equal", adjustable="box")
    ax_nyq.tick_params(axis="both", which="major", labelsize=14, width=1.5, length=0)
    ax_nyq.legend(loc="best", fontsize=15, frameon=False, handlelength=2.2)

    ax_mag.plot(
        freq, mag_exp,
        linestyle="None",
        marker="o",
        markersize=EXP_MARKER_SIZE * 0.85,
        markerfacecolor=EXP_COLOR,
        markeredgecolor=EXP_COLOR,
        label="Experimental",
        zorder=3,
    )
    ax_mag.plot(
        freq_dense, mag_fit,
        linestyle="-",
        linewidth=FIT_LINEWIDTH,
        color=FIT_COLOR,
        label="ECM Fit",
        antialiased=True,
        solid_joinstyle="round",
        solid_capstyle="round",
        zorder=2,
    )

    ax_mag.set_ylabel(
        r"$|\mathbf{Z}|\;(\mathbf{\Omega\cdot cm^{2}})$",
        fontweight="bold",
        fontsize=16,
        labelpad=10
    )

    style_axis(
        ax_mag,
        logx=True,
        logy=True,
        format_log_x=True,
        format_log_y=True,
        use_minor=True
    )

    ax_mag.set_xlim(BODE_XMIN_HZ, np.max(freq_dense))
    ax_mag.set_ylim(BODE_MAG_YMIN_OHM, BODE_MAG_YMAX_OHM)
    ax_mag.tick_params(axis="x", which="both", labelbottom=True)
    ax_mag.legend(loc="best", fontsize=14, frameon=False, handlelength=2.0)

    ax_phs.plot(
        freq, phase_exp,
        linestyle="None",
        marker="o",
        markersize=EXP_MARKER_SIZE * 0.85,
        markerfacecolor=EXP_COLOR,
        markeredgecolor=EXP_COLOR,
        label="Experimental",
        zorder=3,
    )
    ax_phs.plot(
        freq_dense, phase_fit,
        linestyle="-",
        linewidth=FIT_LINEWIDTH,
        color=FIT_COLOR,
        label="ECM Fit",
        antialiased=True,
        solid_joinstyle="round",
        solid_capstyle="round",
        zorder=2,
    )

    ax_phs.set_xlabel(
        r"$f\;(\mathrm{Hz})$",
        fontweight="bold",
        fontsize=16,
        labelpad=8
    )
    ax_phs.set_ylabel(
        r"$-\phi\;(^\circ)$",
        fontweight="bold",
        fontsize=16,
        labelpad=10
    )

    style_axis(
        ax_phs,
        logx=True,
        logy=False,
        format_log_x=True,
        format_log_y=False,
        use_minor=True
    )

    phase_min_data = min(np.nanmin(phase_exp), np.nanmin(phase_fit))
    phase_max_data = max(np.nanmax(phase_exp), np.nanmax(phase_fit))

    phase_lower = PHASE_MAJOR_STEP * np.floor((phase_min_data - 5.0) / PHASE_MAJOR_STEP)
    phase_upper = PHASE_MAJOR_STEP * np.ceil((phase_max_data + 5.0) / PHASE_MAJOR_STEP)

    ax_phs.set_xlim(BODE_XMIN_HZ, np.max(freq_dense))
    ax_phs.set_ylim(phase_lower, phase_upper)
    ax_phs.yaxis.set_major_locator(mticker.MultipleLocator(PHASE_MAJOR_STEP))
    ax_phs.yaxis.set_minor_locator(mticker.MultipleLocator(PHASE_MINOR_STEP))

    for ax in [ax_nyq, ax_mag, ax_phs]:
        ax.tick_params(axis="x", which="major", labelsize=14, width=1.5, pad=8)
        ax.tick_params(axis="y", which="major", labelsize=14, width=1.5, pad=8)

    fig.canvas.draw()
    for ax in [ax_nyq, ax_mag, ax_phs]:
        apply_bold_ticks(ax)
        apply_bold_legend(ax)

    save_figure_png(fig, out_dir / f"{sample_name}_Combined_EIS")
    plt.close(fig)


def plot_residuals(out_dir, sample_name, weighting, freq, z_exp, z_fit):
    res_re = z_fit.real - z_exp.real
    res_im = (-z_fit.imag) - (-z_exp.imag)

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.6), sharex=True)

    axes[0].plot(
        freq, res_re,
        linestyle="None",
        marker="o",
        markersize=6.5,
        markerfacecolor=EXP_COLOR,
        markeredgecolor=EXP_COLOR,
        label=r"$Z^\prime$ residual",
    )
    axes[0].axhline(0, color=FIT_COLOR, linewidth=2.0, linestyle="-")
    axes[0].set_ylabel(r"$\Delta Z^\prime$ ($\Omega\cdot cm^2$)", fontweight="bold", fontsize=16)
    axes[0].set_title(f"Residuals — {sample_name} ({weighting})", fontweight="bold", fontsize=22, pad=10)

    axes[1].plot(
        freq, res_im,
        linestyle="None",
        marker="o",
        markersize=6.5,
        markerfacecolor=FIT_COLOR,
        markeredgecolor=FIT_COLOR,
        label=r"$-Z^{\prime\prime}$ residual",
    )
    axes[1].axhline(0, color=EXP_COLOR, linewidth=2.0, linestyle="-")
    axes[1].set_xlabel(r"$f\;(\mathrm{Hz})$", fontweight="bold", fontsize=16)
    axes[1].set_ylabel(r"$\Delta(-Z^{\prime\prime})$ ($\Omega\cdot cm^2$)", fontweight="bold", fontsize=16)

    for ax in axes:
        style_axis(ax, logx=True, logy=False, format_log_x=True, format_log_y=False, use_minor=True)
        ax.tick_params(axis="both", which="major", labelsize=14, width=1.4)
        ax.set_xlim(BODE_XMIN_HZ, np.max(freq))

    fig.subplots_adjust(left=0.16, right=0.96, top=0.92, bottom=0.11, hspace=0.18)

    fig.canvas.draw()
    for ax in axes:
        apply_bold_ticks(ax)

    save_figure_png(fig, out_dir / f"{sample_name}_Residuals")
    plt.close(fig)


# -----------------------------
# MAIN
# -----------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)

resolved_files = []
for sample_name in PLOT_ORDER:
    candidates = FILE_CANDIDATES[sample_name]
    fpath = resolve_input_file(candidates)
    if fpath is not None:
        resolved_files.append((sample_name, fpath))
    else:
        print(f"[WARNING] Missing file for {sample_name}. Tried: {candidates}")

if len(resolved_files) == 0:
    print("No target CSV files were found.")
    print(f"Checked current working directory: {INPUT_DIR.resolve()}")
    if FALLBACK_INPUT_DIR is not None:
        print(f"Checked fallback directory: {FALLBACK_INPUT_DIR.resolve()}")
    raise FileNotFoundError("None of the uploaded EIS CSV files were found.")

all_summary_rows = []

for weighting in WEIGHTING_SCHEMES:
    print("\n" + "=" * 70)
    print(f"Running weighting scheme: {weighting}")
    print("=" * 70)

    weighting_dir = OUT_DIR / weighting
    weighting_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for sample_name, csv_file in resolved_files:
        trim_high, trim_low, use_robust_loss = get_fit_options(sample_name)

        print(f"Fitting: {csv_file.name}")
        print(
            f"  sample={sample_name}, weighting={weighting}, "
            f"trim_high={trim_high}, trim_low={trim_low}, robust_loss={use_robust_loss}"
        )

        df_all, df_fit, freq, z_exp, n_before, n_after = read_eis_csv(
            csv_file, trim_high, trim_low
        )

        fit_res, stderr, de_res = fit_one_file_hybrid(
            freq, z_exp,
            weighting=weighting,
            use_robust_loss=use_robust_loss
        )

        Rs, R, Q, n = fit_res.x
        z_fit = z_model_rq(freq, Rs, R, Q, n)

        freq_dense = make_dense_frequency(freq, n_points=DENSE_FREQ_POINTS)
        z_fit_dense = z_model_rq(freq_dense, Rs, R, Q, n)

        Ceff = cpe_to_ceff_hsu_mansfeld(R, Q, n)
        f_peak_model = (1.0 / (2.0 * np.pi)) * (1.0 / (R * Q)) ** (1.0 / n)

        rmse_re, rmse_im, wrmse_modulus = fit_metrics(z_exp, z_fit)

        row = {
            "sample": sample_name,
            "source_file": csv_file.name,
            "model": "Rs-(Q||R)",
            "optimizer": "differential_evolution + least_squares",
            "weighting": weighting,
            "n_points_raw": n_before,
            "n_points_used": n_after,
            "trim_high_freq_pts": trim_high,
            "trim_low_freq_pts": trim_low,
            "robust_loss_used": use_robust_loss,
            "f_max_Hz": float(np.max(freq)),
            "f_min_Hz": float(np.min(freq)),
            "Rs_ohm_cm2": Rs,
            "Rs_std": stderr[0],
            "R_ohm_cm2": R,
            "R_std": stderr[1],
            "Q_s^n_per_ohm_cm2": Q,
            "Q_std": stderr[2],
            "n": n,
            "n_std": stderr[3],
            "Ceff_F_cm2": Ceff,
            "f_peak_model_Hz": f_peak_model,
            "RMSE_Re": rmse_re,
            "RMSE_Im": rmse_im,
            "weighted_RMSE_modulus": wrmse_modulus,
            "cost": fit_res.cost,
            "success": bool(fit_res.success),
            "message": fit_res.message,
            "de_fun": de_res.fun,
            "de_nit": de_res.nit,
            "de_nfev": de_res.nfev,
        }

        summary_rows.append(row)
        all_summary_rows.append(row.copy())

        if SAVE_FIT_CSV:
            save_fit_csv(weighting_dir, sample_name, weighting, freq, z_exp, z_fit)

        if SAVE_PLOTS:
            plot_nyquist(weighting_dir, sample_name, weighting, z_exp, z_fit_dense)
            plot_bode(weighting_dir, sample_name, weighting, freq, z_exp, freq_dense, z_fit_dense)

        if SAVE_COMBINED_PLOTS:
            plot_combined_eis_figure(weighting_dir, sample_name, weighting, freq, z_exp, freq_dense, z_fit_dense)

        if SAVE_RESIDUAL_PLOTS:
            plot_residuals(weighting_dir, sample_name, weighting, freq, z_exp, z_fit)

    summary_df = pd.DataFrame(summary_rows).sort_values("sample").reset_index(drop=True)
    summary_df.to_csv(weighting_dir / f"EIS_1RQ_summary_{weighting}.csv", index=False)

    summary_display = summary_df.copy()
    for col in summary_display.columns:
        if pd.api.types.is_numeric_dtype(summary_display[col]):
            summary_display[col] = summary_display[col].map(
                lambda x: f"{x:.5g}" if pd.notna(x) else ""
            )
    summary_display.to_csv(weighting_dir / f"EIS_1RQ_summary_{weighting}_rounded.csv", index=False)

    print(f"\nCompleted weighting = {weighting}")
    print(summary_display.to_string(index=False))

# -----------------------------
# Global comparison table
# -----------------------------
all_summary_df = pd.DataFrame(all_summary_rows)
all_summary_df = all_summary_df.sort_values(["sample", "weighting"]).reset_index(drop=True)
all_summary_df.to_csv(OUT_DIR / "EIS_1RQ_all_weightings_summary.csv", index=False)

comparison = all_summary_df.pivot_table(
    index="sample",
    columns="weighting",
    values="weighted_RMSE_modulus",
    aggfunc="first"
).reset_index()

comparison.columns.name = None

for w in WEIGHTING_SCHEMES:
    if w not in comparison.columns:
        comparison[w] = np.nan

comparison["best_weighting_by_modulus_wRMSE"] = comparison[WEIGHTING_SCHEMES].idxmin(axis=1)
comparison["best_modulus_wRMSE"] = comparison[WEIGHTING_SCHEMES].min(axis=1)

comparison = comparison[
    ["sample"] + WEIGHTING_SCHEMES + ["best_weighting_by_modulus_wRMSE", "best_modulus_wRMSE"]
]
comparison.to_csv(OUT_DIR / "EIS_1RQ_weighting_comparison_table.csv", index=False)

comparison_display = comparison.copy()
for col in comparison_display.columns:
    if pd.api.types.is_numeric_dtype(comparison_display[col]):
        comparison_display[col] = comparison_display[col].map(
            lambda x: f"{x:.5g}" if pd.notna(x) else ""
        )

comparison_display.to_csv(OUT_DIR / "EIS_1RQ_weighting_comparison_table_rounded.csv", index=False)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"All results saved in: {OUT_DIR.resolve()}")
print("\nWeighting comparison:")
print(comparison_display.to_string(index=False))


# In[ ]:




