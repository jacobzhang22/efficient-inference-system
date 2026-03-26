import os
import matplotlib.pyplot as plt
import pandas as pd

from src.config import SchedulingExperimentConfig


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["arrival_rate_rps", "max_batch_size", "batch_timeout_ms"]
    metric_cols = [
        "throughput_rps",
        "throughput_tokens_per_s",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
        "mean_wait_ms",
        "mean_batch_size",
    ]
    return df.groupby(group_cols, as_index=False)[metric_cols].mean()


def _save_line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    line_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: str,
    y_log: bool = False,
    line_label_prefix: str = "",
):
    plt.figure(figsize=(8, 5))

    for value in sorted(df[line_col].unique()):
        sub = df[df[line_col] == value].sort_values(x_col)
        plt.plot(
            sub[x_col],
            sub[y_col],
            marker="o",
            label=f"{line_label_prefix}{value}",
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if y_log:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _save_latency_cdf(
    req_df: pd.DataFrame,
    arrival_rate_rps: float,
    batch_timeout_ms: float,
    batch_sizes: list[int],
    output_path: str,
):
    plt.figure(figsize=(8, 5))

    for batch_size in batch_sizes:
        sub = req_df[
            (req_df["arrival_rate_rps"] == arrival_rate_rps)
            & (req_df["batch_timeout_ms"] == batch_timeout_ms)
            & (req_df["max_batch_size"] == batch_size)
        ].copy()

        if sub.empty:
            continue

        latencies = sub["latency_ms"].sort_values().to_numpy()
        cdf_y = [(i + 1) / len(latencies) for i in range(len(latencies))]

        plt.plot(latencies, cdf_y, label=f"batch={batch_size}")

    plt.xlabel("Request latency (ms)")
    plt.ylabel("CDF")
    plt.title(
        f"Latency CDF at arrival_rate={arrival_rate_rps} req/s, timeout={batch_timeout_ms} ms"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _save_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: str,
    y_log: bool = False,
    point_label_cols: list[str] | None = None,
    group_label_prefix: str = "",
):
    plt.figure(figsize=(8, 5))

    for value in sorted(df[group_col].unique()):
        sub = df[df[group_col] == value].copy()
        plt.scatter(sub[x_col], sub[y_col], label=f"{group_label_prefix}{value}")

        if point_label_cols:
            for _, row in sub.iterrows():
                label = ", ".join(f"{col}={row[col]}" for col in point_label_cols)
                plt.annotate(
                    label,
                    (row[x_col], row[y_col]),
                    fontsize=7,
                    alpha=0.75,
                    xytext=(4, 4),
                    textcoords="offset points",
                )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if y_log:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run():
    cfg = SchedulingExperimentConfig()
    raw_dir = f"{cfg.output_dir}/raw"
    plot_dir = f"{cfg.output_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)

    summary_path = f"{raw_dir}/summary.csv"
    requests_path = f"{raw_dir}/requests.csv"

    if not os.path.exists(summary_path):
        print(f"Missing {summary_path}")
        return

    summary_df = pd.read_csv(summary_path)
    agg = _aggregate(summary_df)

    fixed_timeout = cfg.batch_timeouts_ms[0]

    # Main report plots: fix timeout, compare batch caps across arrival rates.
    timeout_slice = agg[agg["batch_timeout_ms"] == fixed_timeout].copy()

    _save_line_plot(
        df=timeout_slice,
        x_col="arrival_rate_rps",
        y_col="throughput_rps",
        line_col="max_batch_size",
        xlabel="Arrival rate (req/s)",
        ylabel="Throughput (req/s)",
        title=f"Throughput vs arrival rate (timeout={fixed_timeout} ms)",
        output_path=f"{plot_dir}/throughput_vs_arrival_rate_timeout_{fixed_timeout}.png",
        line_label_prefix="batch=",
    )

    _save_line_plot(
        df=timeout_slice,
        x_col="arrival_rate_rps",
        y_col="p99_latency_ms",
        line_col="max_batch_size",
        xlabel="Arrival rate (req/s)",
        ylabel="P99 latency (ms)",
        title=f"P99 latency vs arrival rate (timeout={fixed_timeout} ms)",
        output_path=f"{plot_dir}/p99_latency_vs_arrival_rate_timeout_{fixed_timeout}.png",
        y_log=True,
        line_label_prefix="batch=",
    )

    _save_line_plot(
        df=timeout_slice,
        x_col="arrival_rate_rps",
        y_col="mean_wait_ms",
        line_col="max_batch_size",
        xlabel="Arrival rate (req/s)",
        ylabel="Mean wait time (ms)",
        title=f"Mean wait vs arrival rate (timeout={fixed_timeout} ms)",
        output_path=f"{plot_dir}/mean_wait_vs_arrival_rate_timeout_{fixed_timeout}.png",
        y_log=True,
        line_label_prefix="batch=",
    )

    realized_batch_slice = timeout_slice[timeout_slice["max_batch_size"] > 1].copy()
    _save_line_plot(
        df=realized_batch_slice,
        x_col="arrival_rate_rps",
        y_col="mean_batch_size",
        line_col="max_batch_size",
        xlabel="Arrival rate (req/s)",
        ylabel="Mean realized batch size",
        title=f"Mean realized batch size vs arrival rate (timeout={fixed_timeout} ms)",
        output_path=f"{plot_dir}/mean_realized_batch_size_vs_arrival_rate_timeout_{fixed_timeout}.png",
        line_label_prefix="cap=",
    )

    # Optional but useful: P95 latency too.
    _save_line_plot(
        df=timeout_slice,
        x_col="arrival_rate_rps",
        y_col="p95_latency_ms",
        line_col="max_batch_size",
        xlabel="Arrival rate (req/s)",
        ylabel="P95 latency (ms)",
        title=f"P95 latency vs arrival rate (timeout={fixed_timeout} ms)",
        output_path=f"{plot_dir}/p95_latency_vs_arrival_rate_timeout_{fixed_timeout}.png",
        y_log=True,
        line_label_prefix="batch=",
    )

    # Timeout effect plot: fix the largest batch size and compare windows.
    fixed_batch = max(cfg.max_batch_sizes)
    batch_slice = agg[agg["max_batch_size"] == fixed_batch].copy()

    _save_line_plot(
        df=batch_slice,
        x_col="arrival_rate_rps",
        y_col="mean_wait_ms",
        line_col="batch_timeout_ms",
        xlabel="Arrival rate (req/s)",
        ylabel="Mean wait time (ms)",
        title=f"Mean wait vs arrival rate (batch={fixed_batch})",
        output_path=f"{plot_dir}/mean_wait_vs_arrival_rate_batch_{fixed_batch}_by_timeout.png",
        y_log=True,
        line_label_prefix="timeout=",
    )

    _save_line_plot(
        df=batch_slice,
        x_col="arrival_rate_rps",
        y_col="mean_batch_size",
        line_col="batch_timeout_ms",
        xlabel="Arrival rate (req/s)",
        ylabel="Mean realized batch size",
        title=f"Mean realized batch size vs arrival rate (batch={fixed_batch})",
        output_path=f"{plot_dir}/mean_batch_size_vs_arrival_rate_batch_{fixed_batch}_by_timeout.png",
        line_label_prefix="timeout=",
    )

    # Throughput vs P99 latency frontier using all configs.
    _save_scatter_plot(
        df=agg,
        x_col="throughput_rps",
        y_col="p99_latency_ms",
        group_col="max_batch_size",
        xlabel="Throughput (req/s)",
        ylabel="P99 latency (ms)",
        title="Throughput vs P99 latency across scheduler configurations",
        output_path=f"{plot_dir}/throughput_vs_p99_latency_scatter.png",
        y_log=True,
        point_label_cols=None,
        group_label_prefix="batch=",
    )

    # Request-level latency CDF for a representative high-load regime.
    if os.path.exists(requests_path):
        req_df = pd.read_csv(requests_path)

        representative_arrival_rate = max(cfg.arrival_rates)
        representative_timeout = fixed_timeout
        representative_batches = sorted(cfg.max_batch_sizes)

        _save_latency_cdf(
            req_df=req_df,
            arrival_rate_rps=representative_arrival_rate,
            batch_timeout_ms=representative_timeout,
            batch_sizes=representative_batches,
            output_path=(
                f"{plot_dir}/latency_cdf_arrival_{representative_arrival_rate}"
                f"_timeout_{representative_timeout}.png"
            ),
        )

    print(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    run()