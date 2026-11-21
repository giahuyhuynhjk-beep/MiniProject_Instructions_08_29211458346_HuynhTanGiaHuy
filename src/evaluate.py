"""Evaluation and Benchmarking Module for Tech Article Recommender.
CS467 Advanced AI - Assignment 15 (MP-A08)
Student: Huynh Tan Gia Huy (ID 29211458346)

Implements:
1. Per-user hold-out split (2-3 ratings per user held out, prioritizing relevant rating >= 4).
2. Strict train-only model rebuilding to eliminate data leakage.
3. 20-seed Monte Carlo evaluation reporting Mean +/- Std.
4. Precision@5 across Popularity, Content-Based, User-Based CF, and Hybrid.
5. Rating prediction error (RMSE and MAE) with documented linear mapping for Content-Based.
6. Hyperparameter sweep of alpha in [0.0, 0.25, 0.5, 0.75, 1.0] for HybridRecommender.
7. Publication-quality visualization charts saved to results/ (dpi >= 150).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Add project root to sys.path to enable direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.nlp_pipeline import NLPPipeline
from src.recommender import (
    PopularityRecommender,
    ContentBasedRecommender,
    UserBasedCF,
    HybridRecommender
)
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def split_train_test(
    ratings_df: pd.DataFrame,
    seed: int,
    test_ratings_per_user: int = 2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform per-user hold-out split without cross-user leakage.
    For each user, selects `test_ratings_per_user` items for the test set,
    preferring at least one rating >= 4 (relevant ground truth) if available.
    """
    rng = np.random.RandomState(seed)
    train_rows = []
    test_rows = []

    for user_id, group in ratings_df.groupby("user_id"):
        high_ratings = group[group["rating"] >= 4]
        low_ratings = group[group["rating"] < 4]

        test_indices = []
        if len(high_ratings) > 0:
            # Pick at least 1 high-rating item for test
            chosen_high = rng.choice(high_ratings.index, size=1, replace=False)[0]
            test_indices.append(chosen_high)
            
            # Pick remaining test items from rest of user's ratings
            remaining_indices = [idx for idx in group.index if idx not in test_indices]
            needed = test_ratings_per_user - len(test_indices)
            if remaining_indices and needed > 0:
                chosen_rest = rng.choice(remaining_indices, size=min(needed, len(remaining_indices)), replace=False)
                test_indices.extend(chosen_rest.tolist())
        else:
            # Fallback if no high ratings exist for this user
            test_indices = rng.choice(group.index, size=min(test_ratings_per_user, len(group)), replace=False).tolist()

        test_df_user = group.loc[test_indices]
        train_df_user = group.drop(index=test_indices)

        test_rows.append(test_df_user)
        train_rows.append(train_df_user)

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df


def evaluate_single_split(
    items_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pipeline: NLPPipeline,
    alphas: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    top_k: int = 5
) -> Dict[str, Any]:
    """Train all models solely on train_df and evaluate Precision@K, RMSE, and MAE against test_df."""
    # 1. Initialize models strictly on TRAIN data
    pop_rec = PopularityRecommender(items_df, train_df)
    cb_rec = ContentBasedRecommender(items_df, train_df, pipeline)
    cf_rec = UserBasedCF(items_df, train_df, k_neighbors=5)
    
    hybrid_models = {
        alpha: HybridRecommender(cb_rec, cf_rec, alpha=alpha)
        for alpha in alphas
    }

    # Extract test ground truth per user (relevant if rating >= 4)
    users = sorted(test_df["user_id"].unique())
    user_test_relevant: Dict[str, set] = {}
    user_test_all: Dict[str, Dict[str, float]] = {}
    
    for uid in users:
        u_test = test_df[test_df["user_id"] == uid]
        user_test_all[uid] = dict(zip(u_test["item_id"], u_test["rating"]))
        rel = set(u_test[u_test["rating"] >= 4]["item_id"])
        user_test_relevant[uid] = rel

    # Precision@K accumulators
    precisions: Dict[str, List[float]] = {
        "Popularity": [],
        "Content-Based": [],
        "User-Based CF": []
    }
    for a in alphas:
        precisions[f"Hybrid (alpha={a:.2f})"] = []

    # Evaluation loop per user
    for uid in users:
        rel_set = user_test_relevant[uid]
        if not rel_set:
            continue

        # Exclude seen items from train set
        # Popularity
        pop_recs = pop_rec.recommend_for_user(uid, top_k=top_k, exclude_seen=True)
        pop_hits = sum(1 for item in pop_recs["item_id"] if item in rel_set)
        precisions["Popularity"].append(pop_hits / float(top_k))

        # Content-Based
        cb_recs = cb_rec.recommend_for_user(uid, top_k=top_k, exclude_seen=True)
        cb_hits = sum(1 for item in cb_recs["item_id"] if item in rel_set)
        precisions["Content-Based"].append(cb_hits / float(top_k))

        # User-Based CF
        cf_recs = cf_rec.recommend_for_user(uid, top_k=top_k, exclude_seen=True)
        cf_hits = sum(1 for item in cf_recs["item_id"] if item in rel_set)
        precisions["User-Based CF"].append(cf_hits / float(top_k))

        # Hybrid sweep
        for a in alphas:
            hyb = hybrid_models[a]
            h_recs = hyb.recommend_for_user(uid, top_k=top_k, exclude_seen=True)
            h_hits = sum(1 for item in h_recs["item_id"] if item in rel_set)
            precisions[f"Hybrid (alpha={a:.2f})"].append(h_hits / float(top_k))

    # Error Metrics (RMSE & MAE) on test set pairs
    cf_errors = []
    cb_errors = []
    
    for _, row in test_df.iterrows():
        uid = row["user_id"]
        iid = row["item_id"]
        actual = float(row["rating"])
        
        # CF predicted rating
        cf_pred = cf_rec.predict_rating(uid, iid)
        cf_errors.append(actual - cf_pred)
        
        # Content-based predicted rating:
        # Linear mapping from cosine similarity in [0, 1] to 1-5 rating scale:
        # predicted_rating = 1.0 + 4.0 * cosine_similarity
        user_prof = cb_rec.user_profiles.get(uid, None)
        if user_prof is not None and not np.all(user_prof == 0):
            item_vec = pipeline.get_item_vector(iid)
            sim = float(np.dot(user_prof, item_vec))
            sim = max(0.0, min(1.0, sim))
            cb_pred = 1.0 + 4.0 * sim
        else:
            cb_pred = 3.0
        cb_errors.append(actual - cb_pred)

    cf_rmse = math.sqrt(np.mean(np.square(cf_errors))) if cf_errors else 0.0
    cf_mae = float(np.mean(np.abs(cf_errors))) if cf_errors else 0.0
    cb_rmse = math.sqrt(np.mean(np.square(cb_errors))) if cb_errors else 0.0
    cb_mae = float(np.mean(np.abs(cb_errors))) if cb_errors else 0.0

    result = {
        model_name: float(np.mean(vals))
        for model_name, vals in precisions.items()
    }
    result["CF_RMSE"] = cf_rmse
    result["CF_MAE"] = cf_mae
    result["CB_RMSE"] = cb_rmse
    result["CB_MAE"] = cb_mae
    return result


def run_full_evaluation(
    n_seeds: int = 20,
    base_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Execute evaluation across multiple deterministic random seeds and return summary stats."""
    items_df = pd.read_csv(DATA_DIR / "items.csv")
    ratings_df = pd.read_csv(DATA_DIR / "ratings.csv")

    pipeline = NLPPipeline()
    pipeline.fit_transform(items_df)

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    seed_records: List[Dict[str, Any]] = []

    print(f"Running evaluation across {n_seeds} random seeds (base_seed={base_seed})...")
    for s_idx in range(n_seeds):
        curr_seed = base_seed + s_idx
        train_df, test_df = split_train_test(ratings_df, seed=curr_seed, test_ratings_per_user=2)
        res = evaluate_single_split(items_df, train_df, test_df, pipeline, alphas=alphas, top_k=5)
        res["seed"] = curr_seed
        seed_records.append(res)

    results_df = pd.DataFrame(seed_records)

    # Calculate mean and std for summary table
    metric_cols = [col for col in results_df.columns if col != "seed"]
    summary_data = []
    for col in metric_cols:
        summary_data.append({
            "Metric": col,
            "Mean": results_df[col].mean(),
            "Std": results_df[col].std(),
            "Min": results_df[col].min(),
            "Max": results_df[col].max()
        })
    summary_df = pd.DataFrame(summary_data)

    # Determine best alpha for Hybrid based on Precision@5
    hybrid_alpha_cols = [f"Hybrid (alpha={a:.2f})" for a in alphas]
    best_hybrid_col = max(hybrid_alpha_cols, key=lambda c: results_df[c].mean())
    best_alpha = float(best_hybrid_col.split("=")[1].replace(")", ""))
    print(f"Optimal Hybrid alpha: {best_alpha:.2f} (Highest Mean Precision@5)")

    return results_df, summary_df, best_alpha


def generate_visualizations(
    items_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    best_alpha: float
) -> None:
    """Generate all required high-resolution plots and save them to results/ directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Rating and Topic Distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=180)
    
    # Subplot A: Ratings count distribution
    rating_counts = ratings_df["rating"].value_counts().sort_index()
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
    axes[0].bar(rating_counts.index, rating_counts.values, color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_title("User Rating Distribution (Scale 1 - 5)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Rating Value", fontsize=11)
    axes[0].set_ylabel("Number of Ratings", fontsize=11)
    axes[0].set_xticks([1, 2, 3, 4, 5])
    for x, y in zip(rating_counts.index, rating_counts.values):
        axes[0].text(x, y + 2, str(y), ha="center", fontweight="bold", fontsize=10)

    # Subplot B: Article counts per topic
    topic_counts = items_df["topic"].value_counts()
    axes[1].barh(topic_counts.index, topic_counts.values, color="#3498db", edgecolor="black", alpha=0.85)
    axes[1].set_title("Articles per Domain Topic (45 total)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Article Count", fontsize=11)
    axes[1].set_xlim(0, 7)
    for idx, (val, name) in enumerate(zip(topic_counts.values, topic_counts.index)):
        axes[1].text(val + 0.1, idx, str(val), va="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    chart1_path = RESULTS_DIR / "rating_and_topic_distribution.png"
    plt.savefig(chart1_path, dpi=180)
    plt.close()
    print(f"Saved chart: {chart1_path}")

    # 2. Precision@5 Comparison (with error bars)
    primary_models = [
        "Popularity",
        "Content-Based",
        "User-Based CF",
        f"Hybrid (alpha={best_alpha:.2f})"
    ]
    
    means = [results_df[m].mean() for m in primary_models]
    stds = [results_df[m].std() for m in primary_models]

    plt.figure(figsize=(9, 6), dpi=180)
    bar_colors = ["#95a5a6", "#3498db", "#9b59b6", "#2ecc71"]
    bars = plt.bar(primary_models, means, yerr=stds, capsize=6, color=bar_colors, edgecolor="black", alpha=0.9, width=0.55)
    plt.title("Precision@5 Comparison Across Recommender Architectures\n(20 Random Seeds, Mean +/- Std)", fontsize=13, fontweight="bold")
    plt.ylabel("Precision@5 Score", fontsize=11)
    plt.ylim(0, max(means) * 1.45 if max(means) > 0 else 0.5)

    for bar, mean_val in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            mean_val + 0.015,
            f"{mean_val:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10
        )

    plt.tight_layout()
    chart2_path = RESULTS_DIR / "precision_at_5_comparison.png"
    plt.savefig(chart2_path, dpi=180)
    plt.close()
    print(f"Saved chart: {chart2_path}")

    # 3. Hybrid Alpha Parameter Sweep
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    alpha_cols = [f"Hybrid (alpha={a:.2f})" for a in alphas]
    alpha_means = [results_df[c].mean() for c in alpha_cols]
    alpha_stds = [results_df[c].std() for c in alpha_cols]

    plt.figure(figsize=(8, 5), dpi=180)
    plt.plot(alphas, alpha_means, marker="o", linewidth=2.5, color="#e67e22", label="Precision@5 Mean")
    plt.fill_between(
        alphas,
        np.array(alpha_means) - np.array(alpha_stds),
        np.array(alpha_means) + np.array(alpha_stds),
        color="#e67e22",
        alpha=0.2,
        label="±1 Std Error Band"
    )
    plt.scatter([best_alpha], [results_df[f"Hybrid (alpha={best_alpha:.2f})"].mean()], color="#c0392b", s=100, zorder=5, label=f"Optimal α={best_alpha:.2f}")

    plt.title("Hybrid Recommender: Alpha Parameter Sweep (20 Seeds)", fontsize=13, fontweight="bold")
    plt.xlabel("Alpha (Weight on Content-Based; 1-Alpha on Collaborative Filtering)", fontsize=11)
    plt.ylabel("Precision@5 Score", fontsize=11)
    plt.xticks(alphas)
    plt.legend(frameon=True)
    plt.tight_layout()
    chart3_path = RESULTS_DIR / "hybrid_alpha_sweep.png"
    plt.savefig(chart3_path, dpi=180)
    plt.close()
    print(f"Saved chart: {chart3_path}")

    # 4. User-Item Interaction Sparsity Matrix
    pivot_matrix = ratings_df.pivot(index="user_id", columns="item_id", values="rating")
    sparsity_pct = (1.0 - (ratings_df.shape[0] / (14 * 45))) * 100

    plt.figure(figsize=(14, 5.5), dpi=180)
    plt.imshow(~pivot_matrix.isna(), aspect="auto", cmap="Blues", interpolation="nearest")
    plt.title(f"User-Item Interaction Matrix (Sparsity: {sparsity_pct:.2f}% Missing Ratings)", fontsize=13, fontweight="bold")
    plt.xlabel("Article Index (A001 - A045)", fontsize=11)
    plt.ylabel("User ID (U01 - U14)", fontsize=11)
    plt.yticks(range(len(pivot_matrix.index)), pivot_matrix.index, fontsize=9)
    plt.xticks(range(0, 45, 5), [f"A{i:03d}" for i in range(1, 46, 5)], fontsize=9)
    plt.colorbar(label="Interaction Observed (1=Rated, 0=Unrated)", ticks=[0, 1])
    plt.tight_layout()
    chart4_path = RESULTS_DIR / "user_item_sparsity.png"
    plt.savefig(chart4_path, dpi=180)
    plt.close()
    print(f"Saved chart: {chart4_path}")


if __name__ == "__main__":
    items_df = pd.read_csv(DATA_DIR / "items.csv")
    ratings_df = pd.read_csv(DATA_DIR / "ratings.csv")

    raw_results_df, summary_df, best_alpha = run_full_evaluation(n_seeds=20, base_seed=42)

    # Save evaluation summary to results/
    csv_out = RESULTS_DIR / "evaluation_results.csv"
    summary_df.to_csv(csv_out, index=False)
    print(f"\nSaved evaluation metrics summary to: {csv_out}")
    print("\n=== EVALUATION RESULTS SUMMARY (20 SEEDS) ===")
    print(summary_df.to_string(index=False))

    # Generate visual charts
    generate_visualizations(items_df, ratings_df, raw_results_df, summary_df, best_alpha)
    print("\nAll evaluation charts and CSV outputs generated successfully.")
