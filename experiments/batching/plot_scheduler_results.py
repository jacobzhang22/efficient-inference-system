import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SchedulingExperimentConfig


SUMMARY_GROUP_COLS = [
    "scheduler_mode",
    "arrival_rate_rps",
    "max_batch_size",
    "batch_timeout_ms",
    "scheduling_policy_value",
]

SUMMARY_METRIC_COLS = [
    "throughput_rps",
    "throughput_tokens_per_s",
    "mean_event_tokens_per_s",
    "mean_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "mean_wait_ms",
    "mean_first_token_latency_ms",
    "mean_tokens_scheduled",
    "mean_active_requests",
    "mean_decode_ms_per_token",
    "prefill_runtime_share",
    "decode_runtime_share",
    "mean_live_kv_bytes",
    "mean_reserved_kv_bytes",
    "mean_fragmentation_bytes",
    "mean_workspace_bytes",
    "mean_gpu_allocated_bytes",
    "mean_padding_waste_pct",
    "mean_padding_waste_bytes_est",
    "mean_prompt_len",
    "mean_max_new_tokens",
]


def _mode_order(mode: str) -> int:
    order = {
        "baseline": 0,
        "static": 1,
        "dynamic": 2,
        "continuous": 3,
    }
    return order.get(mode, 99)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(SUMMARY_GROUP_COLS, as_index=False)[SUMMARY_METRIC_COLS].mean()


def _clear_plot_dir(plot_dir: str) -> None:
    os.makedirs(plot_dir, exist_ok=True)
    for filename in os.listdir(plot_dir):
        if filename.endswith(".png"):
            try:
                os.remove(os.path.join(plot_dir, filename))
            except PermissionError:
                # OneDrive placeholder files may deny unlink while still allowing overwrite.
                continue


def _save_multi_mode_plot(
    mode_dfs: dict[str, pd.DataFrame],
    y_col: str,
    ylabel: str,
    title: str,
    output_path: str,
    y_log: bool = False,
) -> None:
    plt.figure(figsize=(8, 5))
    all_batch_sizes: set[int] = set()
    for df in mode_dfs.values():
        all_batch_sizes.update(df["max_batch_size"].unique().tolist())

    line_styles = {
        "dynamic": "--",
        "static": "-.",
        "continuous": "-",
    }
    label_names = {
        "dynamic": "dynamic",
        "static": "static",
        "continuous": "cont",
    }

    for batch_size in sorted(all_batch_sizes):
        for mode, df in sorted(mode_dfs.items(), key=lambda item: _mode_order(item[0])):
            sub = df[df["max_batch_size"] == batch_size].sort_values("arrival_rate_rps")
            if sub.empty:
                continue
            display_name = label_names.get(mode, mode)
            if mode == "dynamic" and batch_size == 1:
                display_name = "baseline"
            plt.plot(
                sub["arrival_rate_rps"],
                sub[y_col],
                marker="o",
                linestyle=line_styles.get(mode, "-"),
                label=f"{display_name},batch={batch_size}",
            )

    plt.xlabel("Arrival rate (req/s)")
    plt.ylabel(ylabel)
    plt.title(title)
    if y_log:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _save_final_family_plot(
    final_dfs: dict[str, pd.DataFrame],
    y_col: str,
    ylabel: str,
    title: str,
    output_path: str,
    y_log: bool = False,
    include_labels: list[str] | None = None,
) -> None:
    plt.figure(figsize=(8, 5))
    line_styles = {
        "baseline": ":",
        "dynamic": "--",
        "static": "-.",
        "continuous": "-",
    }
    label_names = {
        "baseline": "baseline",
        "dynamic": "best dynamic",
        "static": "best static",
        "continuous": "best continuous",
    }

    labels = include_labels or ["baseline", "static", "dynamic", "continuous"]
    for label in labels:
        df = final_dfs.get(label)
        if df is None or df.empty:
            continue
        sub = df.sort_values("arrival_rate_rps")
        plt.plot(
            sub["arrival_rate_rps"],
            sub[y_col],
            marker="o",
            linestyle=line_styles[label],
            label=label_names[label],
        )

    plt.xlabel("Arrival rate (req/s)")
    plt.ylabel(ylabel)
    plt.title(title)
    if y_log:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _save_policy_sweep_plot(
    df: pd.DataFrame,
    scheduler_mode: str,
    y_col: str,
    ylabel: str,
    title: str,
    output_path: str,
    y_log: bool = False,
) -> None:
    plt.figure(figsize=(9, 5))
    sub_df = df[df["scheduler_mode"] == scheduler_mode].copy()
    if sub_df.empty:
        plt.close()
        return

    if scheduler_mode == "dynamic":
        label_col = "batch_timeout_ms"
        label_prefix = "timeout="
    else:
        label_col = "scheduling_policy_value"
        label_prefix = "chunk="

    for batch_size in sorted(sub_df["max_batch_size"].unique()):
        batch_df = sub_df[sub_df["max_batch_size"] == batch_size]
        for label_value in sorted(batch_df[label_col].unique()):
            line = batch_df[batch_df[label_col] == label_value].sort_values("arrival_rate_rps")
            plt.plot(
                line["arrival_rate_rps"],
                line[y_col],
                marker="o",
                label=f"batch={batch_size}, {label_prefix}{label_value}",
            )

    plt.xlabel("Arrival rate (req/s)")
    plt.ylabel(ylabel)
    plt.title(title)
    if y_log:
        plt.yscale("log")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _select_best_policy_rows(df: pd.DataFrame, metric_col: str, minimize: bool) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    sort_cols = [
        "scheduler_mode",
        "arrival_rate_rps",
        "max_batch_size",
        metric_col,
        "throughput_rps",
    ]
    ascending = [True, True, True, minimize, False]
    ranked = df.sort_values(sort_cols, ascending=ascending)
    return ranked.groupby(["scheduler_mode", "arrival_rate_rps", "max_batch_size"], as_index=False).head(1)


def _select_final_family_rows(df: pd.DataFrame, metric_col: str, minimize: bool) -> dict[str, pd.DataFrame]:
    final_rows: dict[str, pd.DataFrame] = {}

    for mode in ["dynamic", "static", "continuous"]:
        mode_df = df[df["scheduler_mode"] == mode].copy()
        if mode_df.empty:
            continue

        if mode == "dynamic":
            candidate_df = mode_df[mode_df["max_batch_size"] > 1].copy()
        else:
            candidate_df = mode_df

        if candidate_df.empty:
            continue

        ranked_configs = (
            candidate_df.groupby(
                ["scheduler_mode", "max_batch_size", "scheduling_policy_value", "batch_timeout_ms"],
                as_index=False,
            )[metric_col]
            .mean()
            .sort_values(metric_col, ascending=minimize)
        )
        best_config = ranked_configs.iloc[0]
        final_rows[mode] = candidate_df[
            (candidate_df["max_batch_size"] == best_config["max_batch_size"])
            & (candidate_df["scheduling_policy_value"] == best_config["scheduling_policy_value"])
            & (candidate_df["batch_timeout_ms"] == best_config["batch_timeout_ms"])
        ].copy()

    dynamic_df = df[df["scheduler_mode"] == "dynamic"].copy()
    baseline_df = dynamic_df[dynamic_df["max_batch_size"] == 1].copy()
    if not baseline_df.empty:
        ranked_baseline = (
            baseline_df.groupby(
                ["scheduler_mode", "max_batch_size", "scheduling_policy_value", "batch_timeout_ms"],
                as_index=False,
            )[metric_col]
            .mean()
            .sort_values(metric_col, ascending=minimize)
        )
        best_baseline = ranked_baseline.iloc[0]
        final_rows["baseline"] = baseline_df[
            (baseline_df["max_batch_size"] == best_baseline["max_batch_size"])
            & (baseline_df["scheduling_policy_value"] == best_baseline["scheduling_policy_value"])
            & (baseline_df["batch_timeout_ms"] == best_baseline["batch_timeout_ms"])
        ].copy()

    return final_rows


def _save_tradeoff_scatter(agg: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(8, 5))
    for mode in sorted(agg["scheduler_mode"].unique(), key=_mode_order):
        sub = agg[agg["scheduler_mode"] == mode]
        label = "baseline(dynamic,batch=1)" if mode == "dynamic" and (sub["max_batch_size"] == 1).any() else mode
        plt.scatter(sub["throughput_rps"], sub["p99_latency_ms"], label=label)
    plt.xlabel("Throughput (req/s)")
    plt.ylabel("P99 latency (ms)")
    plt.yscale("log")
    plt.title("Throughput vs P99 latency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _save_latency_cdf(req_df: pd.DataFrame, arrival_rate_rps: float, output_path: str) -> None:
    plt.figure(figsize=(8, 5))
    for mode in sorted(req_df["scheduler_mode"].unique(), key=_mode_order):
        sub = req_df[req_df["scheduler_mode"] == mode]
        latencies = sub["latency_ms"].sort_values().to_numpy()
        if len(latencies) == 0:
            continue
        cdf_y = [(i + 1) / len(latencies) for i in range(len(latencies))]
        plt.plot(latencies, cdf_y, label=mode)
    plt.xlabel("Request latency (ms)")
    plt.ylabel("CDF")
    plt.title(f"Latency CDF at arrival_rate={arrival_rate_rps} req/s")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run(cfg: SchedulingExperimentConfig | None = None):
    cfg = cfg or SchedulingExperimentConfig()
    raw_dir = f"{cfg.output_dir}/raw"
    plot_dir = f"{cfg.output_dir}/plots"
    _clear_plot_dir(plot_dir)

    summary_path = f"{raw_dir}/summary.csv"
    requests_path = f"{raw_dir}/requests.csv"
    if not os.path.exists(summary_path):
        print(f"Missing {summary_path}")
        return

    summary_df = pd.read_csv(summary_path)
    agg = _aggregate(summary_df)
    _save_policy_sweep_plot(
        agg,
        scheduler_mode="dynamic",
        y_col="throughput_tokens_per_s",
        ylabel="Throughput (tokens/s)",
        title="Dynamic batching tokens/sec policy sweep",
        output_path=f"{plot_dir}/dynamic_throughput_policy_sweep.png",
    )
    _save_policy_sweep_plot(
        agg,
        scheduler_mode="continuous",
        y_col="throughput_tokens_per_s",
        ylabel="Throughput (tokens/s)",
        title="Continuous tokens/sec policy sweep",
        output_path=f"{plot_dir}/continuous_throughput_policy_sweep.png",
    )
    best_throughput = _select_best_policy_rows(agg, metric_col="throughput_tokens_per_s", minimize=False)
    best_p99 = _select_best_policy_rows(agg, metric_col="p99_latency_ms", minimize=True)
    best_first_token = _select_best_policy_rows(agg, metric_col="mean_first_token_latency_ms", minimize=True)
    final_throughput = _select_final_family_rows(agg, metric_col="throughput_tokens_per_s", minimize=False)
    final_p99 = _select_final_family_rows(agg, metric_col="p99_latency_ms", minimize=True)
    final_first_token = _select_final_family_rows(agg, metric_col="mean_first_token_latency_ms", minimize=True)

    _save_multi_mode_plot(
        {
            "dynamic": best_throughput[best_throughput["scheduler_mode"] == "dynamic"],
            "static": best_throughput[best_throughput["scheduler_mode"] == "static"],
            "continuous": best_throughput[best_throughput["scheduler_mode"] == "continuous"],
        },
        y_col="throughput_tokens_per_s",
        ylabel="Throughput (tokens/s)",
        title="Best-policy tokens/sec vs arrival rate",
        output_path=f"{plot_dir}/throughput_mode_comparison.png",
    )
    _save_final_family_plot(
        final_throughput,
        y_col="throughput_tokens_per_s",
        ylabel="Throughput (tokens/s)",
        title="Throughput (tokens/s) vs arrival rate",
        output_path=f"{plot_dir}/throughput_mode_comparison_final.png",
    )
    _save_final_family_plot(
        final_throughput,
        y_col="throughput_rps",
        ylabel="Throughput (req/s)",
        title="Throughput (req/s) vs arrival rate",
        output_path=f"{plot_dir}/req_per_sec_mode_comparison_final.png",
    )
    _save_multi_mode_plot(
        {
            "dynamic": best_p99[best_p99["scheduler_mode"] == "dynamic"],
            "static": best_p99[best_p99["scheduler_mode"] == "static"],
            "continuous": best_p99[best_p99["scheduler_mode"] == "continuous"],
        },
        y_col="p99_latency_ms",
        ylabel="P99 latency (ms)",
        title="Best-policy P99 latency vs arrival rate",
        output_path=f"{plot_dir}/p99_latency_mode_comparison.png",
        y_log=True,
    )
    _save_final_family_plot(
        final_p99,
        y_col="p99_latency_ms",
        ylabel="P99 latency (ms)",
        title="P99 latency (ms) vs arrival rate",
        output_path=f"{plot_dir}/p99_latency_mode_comparison_final.png",
        y_log=True,
    )
    _save_multi_mode_plot(
        {
            "dynamic": best_first_token[best_first_token["scheduler_mode"] == "dynamic"],
            "static": best_first_token[best_first_token["scheduler_mode"] == "static"],
            "continuous": best_first_token[best_first_token["scheduler_mode"] == "continuous"],
        },
        y_col="mean_first_token_latency_ms",
        ylabel="Mean first-token latency (ms)",
        title="Best-policy first-token latency vs arrival rate",
        output_path=f"{plot_dir}/mean_first_token_latency_mode_comparison.png",
        y_log=True,
    )
    _save_final_family_plot(
        final_first_token,
        y_col="mean_first_token_latency_ms",
        ylabel="Mean first-token latency (ms)",
        title="First-token latency (ms) vs arrival rate",
        output_path=f"{plot_dir}/mean_first_token_latency_mode_comparison_final.png",
        y_log=True,
    )
    _save_final_family_plot(
        final_throughput,
        y_col="mean_decode_ms_per_token",
        ylabel="Decode time per token (ms/token)",
        title="Decode time per token (ms/token) vs arrival rate",
        output_path=f"{plot_dir}/decode_ms_per_token_mode_comparison_final.png",
    )
    _save_final_family_plot(
        final_throughput,
        y_col="mean_live_kv_bytes",
        ylabel="Live KV bytes",
        title="Live KV bytes vs arrival rate",
        output_path=f"{plot_dir}/live_kv_bytes_mode_comparison_final.png",
    )
    _save_final_family_plot(
        final_throughput,
        y_col="mean_reserved_kv_bytes",
        ylabel="Reserved KV bytes",
        title="Reserved KV bytes vs arrival rate",
        output_path=f"{plot_dir}/reserved_kv_bytes_mode_comparison_final.png",
    )
    _save_final_family_plot(
        final_throughput,
        y_col="mean_fragmentation_bytes",
        ylabel="Fragmentation bytes",
        title="Fragmentation bytes vs arrival rate",
        output_path=f"{plot_dir}/fragmentation_mode_comparison_final.png",
    )
    _save_final_family_plot(
        final_throughput,
        y_col="prefill_runtime_share",
        ylabel="Prefill runtime share",
        title="Prefill runtime share vs arrival rate",
        output_path=f"{plot_dir}/prefill_runtime_share_mode_comparison_final.png",
    )
    _save_final_family_plot(
        final_throughput,
        y_col="decode_runtime_share",
        ylabel="Decode runtime share",
        title="Decode runtime share vs arrival rate",
        output_path=f"{plot_dir}/decode_runtime_share_mode_comparison_final.png",
    )
    _save_final_family_plot(
        final_throughput,
        y_col="mean_padding_waste_pct",
        ylabel="Mean padding waste (%)",
        title="Padding waste (%) vs arrival rate",
        output_path=f"{plot_dir}/padding_waste_mode_comparison.png",
        include_labels=["static", "dynamic", "continuous"],
    )
    _save_tradeoff_scatter(agg, f"{plot_dir}/throughput_vs_p99_latency_scatter.png")

    if os.path.exists(requests_path):
        req_df = pd.read_csv(requests_path)
        representative_arrival_rate = float(req_df["arrival_rate_rps"].max())
        req_slice = req_df[req_df["arrival_rate_rps"] == representative_arrival_rate].copy()
        _save_latency_cdf(
            req_slice,
            arrival_rate_rps=representative_arrival_rate,
            output_path=f"{plot_dir}/latency_cdf_arrival_{representative_arrival_rate}.png",
        )

    print(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    run()
