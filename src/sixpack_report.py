"""Generate a Minitab-style capability sixpack report."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from statsmodels.stats.diagnostic import normal_ad


@dataclass(frozen=True)
class CapabilitySpecs:
    lsl: float
    usl: float
    target: Optional[float] = None


@dataclass(frozen=True)
class CapabilityStats:
    mean: float
    within_sigma: float
    overall_sigma: float
    cp: float
    cpk: float
    pp: float
    ppk: float
    cpm: float
    ad_stat: float
    ad_p_value: float


def _moving_ranges(values: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(values))


def _within_sigma(moving_ranges: np.ndarray) -> float:
    # Moving range of 2 uses d2 = 1.128 for the unbiasing constant.
    d2 = 1.128
    mr_bar = np.mean(moving_ranges)
    return mr_bar / d2


def _capability_stats(values: np.ndarray, specs: CapabilitySpecs) -> CapabilityStats:
    mean = float(np.mean(values))
    overall_sigma = float(np.std(values, ddof=1))
    mr = _moving_ranges(values)
    within_sigma = float(_within_sigma(mr))

    lsl = specs.lsl
    usl = specs.usl
    target = specs.target if specs.target is not None else (lsl + usl) / 2

    cp = (usl - lsl) / (6 * within_sigma)
    cpk = min((usl - mean) / (3 * within_sigma), (mean - lsl) / (3 * within_sigma))
    pp = (usl - lsl) / (6 * overall_sigma)
    ppk = min((usl - mean) / (3 * overall_sigma), (mean - lsl) / (3 * overall_sigma))
    cpm = (usl - lsl) / (6 * math.sqrt(overall_sigma**2 + (mean - target) ** 2))

    ad_stat, ad_p_value = normal_ad(values)

    return CapabilityStats(
        mean=mean,
        within_sigma=within_sigma,
        overall_sigma=overall_sigma,
        cp=cp,
        cpk=cpk,
        pp=pp,
        ppk=ppk,
        cpm=cpm,
        ad_stat=float(ad_stat),
        ad_p_value=float(ad_p_value),
    )


def _format_sigma(value: float) -> str:
    return f"{value:.3f}"


def _format_capability(value: float) -> str:
    return f"{value:.2f}"


def _plot_i_chart(ax: plt.Axes, values: np.ndarray, cap_stats: CapabilityStats) -> None:
    x = np.arange(1, len(values) + 1)
    ucl = cap_stats.mean + 3 * cap_stats.within_sigma
    lcl = cap_stats.mean - 3 * cap_stats.within_sigma

    ax.plot(x, values, color="#1f77b4", marker="o", markersize=3, linewidth=0.8)
    ax.axhline(cap_stats.mean, color="#2ca02c", linewidth=1)
    ax.axhline(ucl, color="#d62728", linewidth=1)
    ax.axhline(lcl, color="#d62728", linewidth=1)

    ax.set_title("I Chart", fontsize=10, pad=10, fontweight="bold")
    ax.set_ylabel("Individual Value", fontsize=9)
    ax.set_xlim(1, len(values))
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.set_facecolor("white")

    ax.text(
        1.02,
        0.9,
        f"UCL={ucl:.3f}",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.text(
        1.02,
        0.5,
        f"$\\bar{{X}}$={cap_stats.mean:.3f}",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.text(
        1.02,
        0.1,
        f"LCL={lcl:.3f}",
        transform=ax.transAxes,
        fontsize=8,
    )


def _plot_mr_chart(ax: plt.Axes, values: np.ndarray) -> None:
    mr = _moving_ranges(values)
    mr_bar = np.mean(mr)
    d3, d4 = 0.0, 3.267
    ucl = d4 * mr_bar
    lcl = d3 * mr_bar

    x = np.arange(2, len(values) + 1)
    ax.plot(x, mr, color="#1f77b4", marker="o", markersize=3, linewidth=0.8)
    ax.axhline(mr_bar, color="#2ca02c", linewidth=1)
    ax.axhline(ucl, color="#d62728", linewidth=1)
    ax.axhline(lcl, color="#d62728", linewidth=1)

    ax.set_title("Moving Range Chart", fontsize=10, pad=10, fontweight="bold")
    ax.set_ylabel("Moving Range", fontsize=9)
    ax.set_xlim(1, len(values))
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.set_facecolor("white")

    ax.text(
        1.02,
        0.85,
        f"UCL={ucl:.3f}",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.text(
        1.02,
        0.5,
        f"$\\bar{{M}}$R={mr_bar:.3f}",
        transform=ax.transAxes,
        fontsize=8,
    )
    ax.text(
        1.02,
        0.15,
        "LCL=0",
        transform=ax.transAxes,
        fontsize=8,
    )


def _plot_last_observations(ax: plt.Axes, values: np.ndarray, cap_stats: CapabilityStats) -> None:
    last_values = values[-25:]
    x = np.arange(len(values) - len(last_values) + 1, len(values) + 1)
    ax.scatter(x, last_values, color="#1f77b4", s=16)
    ax.axhline(cap_stats.mean, color="#2ca02c", linewidth=1, linestyle="--")
    ax.set_title("Last 25 Observations", fontsize=10, pad=10, fontweight="bold")
    ax.set_xlabel("Observation", fontsize=9)
    ax.set_ylabel("Values", fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.set_facecolor("white")


def _plot_histogram(
    ax: plt.Axes,
    values: np.ndarray,
    cap_stats: CapabilityStats,
    specs: CapabilitySpecs,
) -> None:
    _counts, _bins, _ = ax.hist(
        values,
        bins=12,
        color="#8cb4e2",
        edgecolor="#4c72b0",
        alpha=0.7,
        density=True,
    )

    x = np.linspace(min(values) - 1, max(values) + 1, 200)
    ax.plot(
        x,
        scipy_stats.norm.pdf(x, cap_stats.mean, cap_stats.within_sigma),
        color="#d62728",
        linewidth=1.2,
        label="Within",
    )
    ax.plot(
        x,
        scipy_stats.norm.pdf(x, cap_stats.mean, cap_stats.overall_sigma),
        color="#1f77b4",
        linewidth=1.2,
        label="Overall",
    )

    ax.axvline(specs.lsl, color="#d62728", linestyle="--", linewidth=1)
    ax.axvline(specs.usl, color="#d62728", linestyle="--", linewidth=1)
    if specs.target is not None:
        ax.axvline(specs.target, color="#2ca02c", linestyle=":", linewidth=1)

    ax.set_title("Capability Histogram", fontsize=10, pad=12, fontweight="bold")
    ax.set_xlabel("Values", fontsize=9)
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.set_facecolor("white")

    y_min, y_max = ax.get_ylim()
    label_y = y_max * 1.02
    ax.text(
        specs.lsl,
        label_y,
        "LSL",
        color="#d62728",
        fontsize=8,
        ha="center",
        va="bottom",
        clip_on=False,
    )
    if specs.target is not None:
        ax.text(
            specs.target,
            label_y,
            "Target",
            color="#2ca02c",
            fontsize=8,
            ha="center",
            va="bottom",
            clip_on=False,
        )
    ax.text(
        specs.usl,
        label_y,
        "USL",
        color="#d62728",
        fontsize=8,
        ha="center",
        va="bottom",
        clip_on=False,
    )

    spec_text = "\n".join(
        [
            "Specifications",
            f"LSL  {specs.lsl:.0f}",
            f"Target  {specs.target:.0f}" if specs.target is not None else "",
            f"USL  {specs.usl:.0f}",
        ]
    ).strip()
    info_x = 1.05
    legend = ax.legend(
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(info_x, 1.08),
        borderaxespad=0.0,
        frameon=True,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#cccccc")
    legend.get_frame().set_linewidth(1)

    ax.text(
        info_x,
        0.7,
        spec_text,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
        clip_on=False,
    )


def _plot_probability(ax: plt.Axes, values: np.ndarray, cap_stats: CapabilityStats) -> None:
    (osm, osr), (slope, intercept, _r) = scipy_stats.probplot(values, dist="norm")
    ax.scatter(osr, osm, color="#1f77b4", s=12)
    fit = slope * np.array(osm) + intercept
    ax.plot(fit, osm, color="#d62728", linewidth=1)

    ax.set_title("Normal Prob Plot", fontsize=10, pad=10, fontweight="bold")
    ax.set_xlabel("Observed", fontsize=9)
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_facecolor("white")

    ax.text(
        0.98,
        0.1,
        f"AD: {cap_stats.ad_stat:.3f}, P: {cap_stats.ad_p_value:.3f}",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
    )


def _plot_capability(ax: plt.Axes, cap_stats: CapabilityStats, specs: CapabilitySpecs) -> None:
    ax.set_title("Capability Plot", fontsize=10, pad=10, fontweight="bold")
    ax.set_xlim(specs.lsl - 1, specs.usl + 1)
    ax.set_ylim(0, 3)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["Specs", "Within", "Overall"], fontsize=8)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax.set_facecolor("white")

    def _interval(y: float, center: float, sigma: float, color: str) -> None:
        width = 3 * sigma
        start = center - width
        end = center + width
        ax.hlines(y, start, end, color=color, linewidth=2)
        ax.plot(
            [start, center, end],
            [y, y, y],
            linestyle="None",
            marker="+",
            color=color,
            markersize=9,
            markeredgewidth=1.6,
        )

    _interval(2.5, cap_stats.mean, cap_stats.overall_sigma, "#1f77b4")
    _interval(1.5, cap_stats.mean, cap_stats.within_sigma, "#d62728")

    ax.hlines(0.5, specs.lsl, specs.usl, color="#2ca02c", linewidth=2)
    ax.plot(
        [specs.lsl, (specs.lsl + specs.usl) / 2, specs.usl],
        [0.5, 0.5, 0.5],
        linestyle="None",
        marker="+",
        color="#2ca02c",
        markersize=9,
        markeredgewidth=1.6,
    )

    ax.set_xlabel("Values", fontsize=9)

    stats_text = "\n".join(
        [
            "Within",
            f"StDev  {_format_sigma(cap_stats.within_sigma)}",
            f"Cp  {_format_capability(cap_stats.cp)}",
            f"Cpk  {_format_capability(cap_stats.cpk)}",
            "",
            "Overall",
            f"StDev  {_format_sigma(cap_stats.overall_sigma)}",
            f"Pp  {_format_capability(cap_stats.pp)}",
            f"Ppk  {_format_capability(cap_stats.ppk)}",
            f"Cpm  {_format_capability(cap_stats.cpm)}",
        ]
    )
    ax.text(
        1.02,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc"),
    )


def generate_sixpack(
    values: Iterable[float],
    specs: CapabilitySpecs,
    title: str,
    output_path: Path,
) -> CapabilityStats:
    values_array = np.asarray(list(values), dtype=float)
    if values_array.size < 5:
        raise ValueError("Need at least 5 observations for sixpack.")

    stats = _capability_stats(values_array, specs)

    fig = plt.figure(figsize=(10, 7), dpi=150)
    fig.suptitle(title, fontsize=12, y=0.98)
    fig.patch.set_facecolor("#efefef")
    grid = fig.add_gridspec(3, 2, hspace=0.6, wspace=0.35)

    _plot_i_chart(fig.add_subplot(grid[0, 0]), values_array, stats)
    _plot_histogram(fig.add_subplot(grid[0, 1]), values_array, stats, specs)
    _plot_mr_chart(fig.add_subplot(grid[1, 0]), values_array)
    _plot_probability(fig.add_subplot(grid[1, 1]), values_array, stats)
    _plot_last_observations(fig.add_subplot(grid[2, 0]), values_array, stats)
    _plot_capability(fig.add_subplot(grid[2, 1]), stats, specs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return stats


def _synthetic_data(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=104.6, scale=1.1, size=100)
    drift = np.linspace(-0.3, 0.3, num=100)
    return data + drift


def main() -> None:
    values = _synthetic_data()
    specs = CapabilitySpecs(lsl=103.0, usl=110.0, target=104.0)
    output = Path("output/sixpack_report.png")

    stats = generate_sixpack(values, specs, "Process Capability Sixpack Report for data", output)

    print(f"Sixpack saved to {output.resolve()}")
    print(
        "Capability summary:",
        f"Mean={stats.mean:.3f}",
        f"Within Sigma={stats.within_sigma:.3f}",
        f"Overall Sigma={stats.overall_sigma:.3f}",
    )


if __name__ == "__main__":
    main()
