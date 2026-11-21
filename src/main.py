"""Command-Line Interface and Demonstration Suite for Tech Article Recommender.
CS467 Advanced AI - Assignment 15 (MP-A08)
Student: Huynh Tan Gia Huy (ID 29211458346)

Supports:
- --demo: Executes all 3 required test cases, displays outputs, and writes results/sample_outputs.txt.
- --user <USER_ID> [--method popularity|content|cf|hybrid]: Generates recommendations for a specific user.
- --query "<TEXT>": Semantic query search across articles.
- --new-article "<TEXT>": Cold-start recommendation (similar articles + likely readers + analytical explanation).
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def format_item_output(rank: int, item_id: str, headline: str, topic: str, score: float, keywords: str) -> str:
    """Format single recommended item output strictly adhering to project specifications."""
    return f"Rank {rank}: [{item_id}] ({topic}) {headline}\n        Score: {score:.4f} | Matched keywords: {keywords}"


def print_and_capture(text: str, stream: io.StringIO) -> None:
    """Print text to stdout and simultaneously capture it in an in-memory stream."""
    print(text)
    stream.write(text + "\n")


def run_demo(items_df: pd.DataFrame, ratings_df: pd.DataFrame, pipeline: NLPPipeline) -> str:
    """Run the 3 mandatory evaluation test cases and format output for console and sample_outputs.txt."""
    output_stream = io.StringIO()

    p = lambda text="": print_and_capture(text, output_stream)

    p("=" * 80)
    p("CS467 ADVANCED AI - ASSIGNMENT 15 (MP-A08)")
    p("Tech Blog and News Article Recommender Demonstration")
    p("Student: Huỳnh Tấn Gia Huy | ID: 29211458346 | Section: CS467A (No. 08)")
    p("=" * 80)

    # Initialize Recommenders on full training/interaction data
    pop_rec = PopularityRecommender(items_df, ratings_df)
    cb_rec = ContentBasedRecommender(items_df, ratings_df, pipeline)
    cf_rec = UserBasedCF(items_df, ratings_df, k_neighbors=5)
    hybrid_rec = HybridRecommender(cb_rec, cf_rec, alpha=0.75)  # optimal alpha from evaluate.py

    # -------------------------------------------------------------
    # TEST CASE 1: Reader Profile (U01 - Cybersecurity & Linux Fan)
    # -------------------------------------------------------------
    p("\n" + "#" * 80)
    p("TEST CASE 1: Reader Profile - U01 (Cybersecurity & Linux Specialist)")
    p("#" * 80)
    
    u01_profile = pipeline.build_user_profile("U01", ratings_df)
    top_kw = pipeline.get_top_user_keywords(u01_profile, top_n=6)
    kw_str = ", ".join([f"{term} ({weight:.3f})" for term, weight in top_kw])
    p(f"User ID: U01")
    p(f"Known Taste: Enthusiast of Cybersecurity and Linux systems.")
    p(f"Learned Profile Keywords: {kw_str}")
    
    p("\n[1.1] Content-Based Top-5 Unread Recommendations for U01:")
    cb_recs = cb_rec.recommend_for_user("U01", top_k=5, exclude_seen=True)
    for idx, row in cb_recs.iterrows():
        p(format_item_output(idx + 1, row["item_id"], row["headline"], row["topic"], row["score"], row["keywords"]))

    p("\n[1.2] Model Comparison Top-5 Unread for U01:")
    p("  -- Popularity Recommender (Bayesian Average):")
    pop_recs = pop_rec.recommend_for_user("U01", top_k=5, exclude_seen=True)
    for idx, row in pop_recs.iterrows():
        p(f"     Rank {idx+1}: [{row['item_id']}] ({row['topic']}) {row['headline'][:60]}... | Score: {row['score']:.4f}")

    p("\n  -- User-Based Collaborative Filtering (Mean-Centered Cosine, k=5):")
    cf_recs = cf_rec.recommend_for_user("U01", top_k=5, exclude_seen=True)
    for idx, row in cf_recs.iterrows():
        p(f"     Rank {idx+1}: [{row['item_id']}] ({row['topic']}) {row['headline'][:60]}... | Score: {row['score']:.4f}")

    p("\n  -- Hybrid Recommender (Optimal alpha=0.75):")
    hyb_recs = hybrid_rec.recommend_for_user("U01", top_k=5, exclude_seen=True)
    for idx, row in hyb_recs.iterrows():
        p(f"     Rank {idx+1}: [{row['item_id']}] ({row['topic']}) {row['headline'][:60]}... | Score: {row['score']:.4f}")

    # -------------------------------------------------------------
    # TEST CASE 2: Free-Form Query Search
    # -------------------------------------------------------------
    p("\n" + "#" * 80)
    p("TEST CASE 2: Query Search - 'open source large language models on laptop'")
    p("#" * 80)
    query_text = "open source large language models on laptop"
    p(f"Search Query: \"{query_text}\"")
    p("Top-5 Matching Articles:")
    
    query_recs = cb_rec.recommend_for_query(query_text, top_k=5)
    for idx, row in query_recs.iterrows():
        p(format_item_output(idx + 1, row["item_id"], row["headline"], row["topic"], row["score"], row["keywords"]))

    # -------------------------------------------------------------
    # TEST CASE 3: Cold-Start New Article
    # -------------------------------------------------------------
    p("\n" + "#" * 80)
    p("TEST CASE 3: Cold-Start Unrated New Article")
    p("#" * 80)
    new_headline = "Critical zero-day vulnerability discovered in the Linux kernel allows privilege escalation; maintainers rush emergency patch"
    new_summary = (
        "Security researchers uncovered a critical zero-day memory vulnerability inside the Linux kernel allowing "
        "unprivileged local users to achieve root execution. Kernel maintainers have released an emergency security patch "
        "and advised rapid deployment across enterprise Linux server fleets."
    )
    new_tags = "linux;kernel;vulnerability;zero-day;privilege-escalation;patch;security"

    p(f"New Article Headline: \"{new_headline}\"")
    p(f"New Article Summary:  \"{new_summary}\"")
    p(f"New Article Tags:     \"{new_tags}\"")
    
    sim_articles, top_readers = cb_rec.recommend_for_new_article(
        headline=new_headline,
        summary_text=new_summary,
        tags=new_tags,
        top_k=5,
        top_users_k=3
    )

    p("\n[3.1] Top-5 Most Similar Existing Articles in Catalog:")
    for idx, row in sim_articles.iterrows():
        p(format_item_output(idx + 1, row["item_id"], row["headline"], row["topic"], row["score"], row["keywords"]))

    p("\n[3.2] Top-3 Likely Readers (User Profile Cosine Similarity):")
    for idx, row in top_readers.iterrows():
        p(f"Rank {idx+1}: User [{row['user_id']}] | Similarity Score: {row['score']:.4f} | Matched keywords: {row['keywords']}")

    p("\n[3.3] Analytical Cold-Start Explanation:")
    p("  * Why Collaborative Filtering Fails: User-Based CF relies on existing user-item interaction matrices.")
    p("    A brand-new article has zero ratings from any user, creating an empty rating vector (cold item problem).")
    p("    Consequently, CF cannot compute neighborhood similarities and yields an undefined prediction.")
    p("  * Why Popularity Baseline Fails: Popularity calculates Bayesian average based on historical ratings.")
    p("    With v=0 ratings, it falls back strictly to the global mean (C=3.0) and cannot differentiate or rank the new item.")
    p("  * Why Content-Based NLP Succeeds: Content-based vectorization operates purely in the semantic text space.")
    p("    It computes TF-IDF representations directly from the headline, summary, and tags, enabling immediate")
    p("    cosine similarity matching against both existing articles and accumulated reader taste profiles.")

    p("\n" + "=" * 80)
    p("DEMONSTRATION COMPLETE - ALL 3 TEST CASES EXECUTED SUCCESSFULLY")
    p("=" * 80)

    captured_text = output_stream.getvalue()
    
    # Save output to results/sample_outputs.txt
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sample_out_file = RESULTS_DIR / "sample_outputs.txt"
    with open(sample_out_file, "w", encoding="utf-8") as f:
        f.write(captured_text)
    print(f"\n[INFO] Complete test case output written to: {sample_out_file}")

    return captured_text


def main():
    parser = argparse.ArgumentParser(
        description="Tech Blog and News Article Recommender CLI (CS467 MP-A08)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--demo", action="store_true", help="Execute the 3 comprehensive test cases and write results/sample_outputs.txt")
    parser.add_argument("--user", type=str, default=None, help="Target user ID (e.g., U01, U02, ..., U14)")
    parser.add_argument(
        "--method",
        type=str,
        default="hybrid",
        choices=["popularity", "content", "cf", "hybrid"],
        help="Recommendation method to employ (default: hybrid)"
    )
    parser.add_argument("--query", type=str, default=None, help="Free-form natural language query string")
    parser.add_argument("--new-article", type=str, default=None, help="Text headline/summary for a new cold-start article")
    parser.add_argument("--top-k", type=int, default=5, help="Number of items to recommend (default: 5)")

    args = parser.parse_args()

    # Load datasets
    items_path = DATA_DIR / "items.csv"
    ratings_path = DATA_DIR / "ratings.csv"
    
    if not items_path.exists() or not ratings_path.exists():
        sys.exit(f"Error: Required datasets missing from {DATA_DIR}. Run data/generate_data.py first.")

    items_df = pd.read_csv(items_path)
    ratings_df = pd.read_csv(ratings_path)

    # Initialize and fit NLP pipeline
    pipeline = NLPPipeline()
    pipeline.fit_transform(items_df)

    # Default action: run demo if --demo or no arguments provided
    if args.demo or (args.user is None and args.query is None and args.new_article is None):
        run_demo(items_df, ratings_df, pipeline)
        return

    # User recommendation
    if args.user is not None:
        user_id = args.user.upper().strip()
        all_users = set(ratings_df["user_id"].unique())
        if user_id not in all_users:
            sys.exit(f"Error: Unknown user_id '{user_id}'. Available users: {sorted(list(all_users))}")

        print(f"\nGenerating Top-{args.top_k} recommendations for {user_id} using method: '{args.method}'")
        if args.method == "popularity":
            rec = PopularityRecommender(items_df, ratings_df)
            res = rec.recommend_for_user(user_id, top_k=args.top_k)
        elif args.method == "content":
            rec = ContentBasedRecommender(items_df, ratings_df, pipeline)
            res = rec.recommend_for_user(user_id, top_k=args.top_k)
        elif args.method == "cf":
            rec = UserBasedCF(items_df, ratings_df, k_neighbors=5)
            res = rec.recommend_for_user(user_id, top_k=args.top_k)
        else:  # hybrid
            cb_rec = ContentBasedRecommender(items_df, ratings_df, pipeline)
            cf_rec = UserBasedCF(items_df, ratings_df, k_neighbors=5)
            rec = HybridRecommender(cb_rec, cf_rec, alpha=0.75)
            res = rec.recommend_for_user(user_id, top_k=args.top_k)

        for idx, row in res.iterrows():
            print(format_item_output(idx + 1, row["item_id"], row["headline"], row["topic"], row["score"], row["keywords"]))
        return

    # Semantic Query Search
    if args.query is not None:
        print(f"\nSearching articles for query: \"{args.query}\"")
        cb_rec = ContentBasedRecommender(items_df, ratings_df, pipeline)
        res = cb_rec.recommend_for_query(args.query, top_k=args.top_k)
        for idx, row in res.iterrows():
            print(format_item_output(idx + 1, row["item_id"], row["headline"], row["topic"], row["score"], row["keywords"]))
        return

    # Cold-Start New Article
    if args.new_article is not None:
        print(f"\nProcessing cold-start new article: \"{args.new_article}\"")
        cb_rec = ContentBasedRecommender(items_df, ratings_df, pipeline)
        sim_items, top_readers = cb_rec.recommend_for_new_article(
            headline=args.new_article,
            summary_text=args.new_article,
            tags="",
            top_k=args.top_k,
            top_users_k=3
        )
        print("\nTop Similar Articles in Catalog:")
        for idx, row in sim_items.iterrows():
            print(format_item_output(idx + 1, row["item_id"], row["headline"], row["topic"], row["score"], row["keywords"]))
        print("\nTop Likely Readers:")
        for idx, row in top_readers.iterrows():
            print(f"Rank {idx+1}: User [{row['user_id']}] | Similarity: {row['score']:.4f} | Matched keywords: {row['keywords']}")
        return


if __name__ == "__main__":
    main()
