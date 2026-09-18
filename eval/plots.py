"""
Evaluation Results Plotting Module
==================================
Generates high-resolution visualization charts from results/metrics.csv:
  1. results/plots/recall_k.png
  2. results/plots/rouge_l.png
  3. results/plots/mrr_chart.png
  4. results/plots/citation_coverage.png
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Use non-GUI headless backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

METRICS_CSV = Path("results/metrics.csv")
PLOTS_DIR = Path("results/plots")

# Set theme styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def generate_all_plots(csv_path: Path = METRICS_CSV, out_dir: Path = PLOTS_DIR) -> None:
    """Generate all 4 evaluation figures from metrics.csv."""
    if not csv_path.exists():
        logger.error(f"Metrics file not found: {csv_path}. Run evaluate.py first.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded metrics dataframe ({len(df)} rows)")

    # ─── 1. Recall@K Grouped Bar Chart ───────────────────────────
    plt.figure(figsize=(10, 5.5))
    recall_cols = ["recall@1", "recall@5", "recall@10"]
    melted = df.melt(
        id_vars=["config", "retrieval_method", "embedding_model"],
        value_vars=recall_cols,
        var_name="Metric",
        value_name="Score",
    )
    
    ax1 = sns.barplot(
        data=melted,
        x="config",
        y="Score",
        hue="Metric",
        palette=["#3B82F6", "#10B981", "#6366F1"],
    )
    plt.title("Retrieval Performance: Recall@1, Recall@5, Recall@10 across Configurations", pad=15)
    plt.xlabel("Evaluation Configuration")
    plt.ylabel("Recall Score")
    plt.ylim(0.0, 1.05)
    plt.legend(title="Metric", loc="upper left")
    plt.tight_layout()
    plot1_path = out_dir / "recall_k.png"
    plt.savefig(plot1_path)
    plt.close()
    logger.info(f"Saved: {plot1_path}")

    # ─── 2. ROUGE-L F1 Generation Score ──────────────────────────
    plt.figure(figsize=(9, 5))
    ax2 = sns.barplot(
        data=df,
        x="config",
        y="rouge_l_f1",
        hue="retrieval_method",
        dodge=False,
        palette="viridis",
    )
    plt.title("ROUGE-L F1 Answer Generation Accuracy vs Reference Answers", pad=15)
    plt.xlabel("Evaluation Configuration")
    plt.ylabel("ROUGE-L F1 Score")
    plt.ylim(0.0, 1.0)
    for p in ax2.patches:
        height = p.get_height()
        if height > 0:
            ax2.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height + 0.02),
                         ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plot2_path = out_dir / "rouge_l.png"
    plt.savefig(plot2_path)
    plt.close()
    logger.info(f"Saved: {plot2_path}")

    # ─── 3. MRR@10 Comparison ────────────────────────────────────
    plt.figure(figsize=(9, 5))
    ax3 = sns.barplot(
        data=df,
        x="config",
        y="mrr@10",
        hue="embedding_model",
        palette=["#0EA5E9", "#F59E0B"],
    )
    plt.title("Mean Reciprocal Rank (MRR@10) by Retrieval Configuration", pad=15)
    plt.xlabel("Evaluation Configuration")
    plt.ylabel("MRR@10 Score")
    plt.ylim(0.0, 1.05)
    for p in ax3.patches:
        height = p.get_height()
        if height > 0:
            ax3.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height + 0.02),
                         ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plot3_path = out_dir / "mrr_chart.png"
    plt.savefig(plot3_path)
    plt.close()
    logger.info(f"Saved: {plot3_path}")

    # ─── 4. Citation Coverage (%) ────────────────────────────────
    plt.figure(figsize=(9, 5))
    ax4 = sns.barplot(
        data=df,
        x="config",
        y="citation_coverage",
        color="#10B981",
    )
    plt.title("Factual Citation Coverage (%) across Evaluated Configurations", pad=15)
    plt.xlabel("Evaluation Configuration")
    plt.ylabel("Citation Coverage (Ratio)")
    plt.ylim(0.0, 1.1)
    for p in ax4.patches:
        height = p.get_height()
        if height > 0:
            ax4.annotate(f"{height:.0%}", (p.get_x() + p.get_width() / 2., height + 0.02),
                         ha='center', va='bottom', fontsize=9, fontweight="bold")
    plt.tight_layout()
    plot4_path = out_dir / "citation_coverage.png"
    plt.savefig(plot4_path)
    plt.close()
    logger.info(f"Saved: {plot4_path}")


if __name__ == "__main__":
    generate_all_plots()
