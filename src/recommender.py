"""Recommender Systems Engine for Tech Blog and News Article Recommender.
CS467 Advanced AI - Assignment 15 (MP-A08)
Student: Huynh Tan Gia Huy (ID 29211458346)

Implements:
1. PopularityRecommender (Bayesian average with minimum vote threshold)
2. ContentBasedRecommender (TF-IDF cosine similarity, query/item/user/cold-start)
3. UserBasedCF (Mean-centered collaborative filtering with k-NN cosine similarity)
4. HybridRecommender (Linear combination of min-max normalized Content and CF scores)
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.nlp_pipeline import NLPPipeline, preprocess


def _format_keywords(keywords: List[str]) -> str:
    """Format list of keyword strings into a comma-separated string."""
    return ", ".join(keywords) if keywords else "n/a"


class PopularityRecommender:
    """Popularity-based baseline using Bayesian average rating:
    score = (v / (v + m)) * R + (m / (v + m)) * C
    where v = number of ratings, m = prior weight threshold,
          R = item average rating, C = global average rating.
    """

    def __init__(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame, min_votes: int = 3):
        self.items_df = items_df.copy().reset_index(drop=True)
        self.ratings_df = ratings_df.copy().reset_index(drop=True)
        self.min_votes = min_votes
        
        # Calculate Bayesian average for each item
        self.global_mean: float = float(self.ratings_df["rating"].mean()) if not self.ratings_df.empty else 3.0
        
        # Aggregate item statistics
        stats = self.ratings_df.groupby("item_id").agg(
            v=("rating", "count"),
            R=("rating", "mean")
        ).reset_index()
        
        m = float(self.min_votes)
        C = self.global_mean
        stats["score"] = stats.apply(
            lambda row: (row["v"] / (row["v"] + m)) * row["R"] + (m / (row["v"] + m)) * C,
            axis=1
        )
        
        # Merge with items metadata
        self.item_scores = self.items_df.merge(stats, on="item_id", how="left")
        self.item_scores["v"] = self.item_scores["v"].fillna(0)
        self.item_scores["R"] = self.item_scores["R"].fillna(C)
        self.item_scores["score"] = self.item_scores["score"].fillna(C)
        self.item_scores = self.item_scores.sort_values(by="score", ascending=False).reset_index(drop=True)

    def recommend_global(self, top_k: int = 5) -> pd.DataFrame:
        """Return globally most popular articles."""
        top_items = self.item_scores.head(top_k).copy()
        top_items["keywords"] = top_items["tags"].apply(lambda t: ", ".join(str(t).split(";")[:3]))
        return top_items[["item_id", "headline", "topic", "score", "keywords"]]

    def recommend_for_user(self, user_id: str, top_k: int = 5, exclude_seen: bool = True) -> pd.DataFrame:
        """Recommend popular articles for a specific user, optionally excluding already-seen articles."""
        seen_items = set()
        if exclude_seen and not self.ratings_df.empty:
            seen_items = set(self.ratings_df[self.ratings_df["user_id"] == user_id]["item_id"])
            
        candidates = self.item_scores[~self.item_scores["item_id"].isin(seen_items)].copy()
        top_items = candidates.head(top_k).copy()
        top_items["keywords"] = top_items["tags"].apply(lambda t: ", ".join(str(t).split(";")[:3]))
        return top_items[["item_id", "headline", "topic", "score", "keywords"]]


class ContentBasedRecommender:
    """TF-IDF Content-Based Recommender using cosine similarity over article documents."""

    def __init__(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame, pipeline: NLPPipeline):
        self.items_df = items_df.copy().reset_index(drop=True)
        self.ratings_df = ratings_df.copy().reset_index(drop=True)
        self.pipeline = pipeline
        self.user_profiles = self.pipeline.build_all_user_profiles(self.ratings_df)

    def recommend_for_user(self, user_id: str, top_k: int = 5, exclude_seen: bool = True) -> pd.DataFrame:
        """Recommend items for user_id based on cosine similarity with their profile vector."""
        if user_id not in self.user_profiles:
            # Try to build profile on the fly
            profile = self.pipeline.build_user_profile(user_id, self.ratings_df)
            self.user_profiles[user_id] = profile
        else:
            profile = self.user_profiles[user_id]

        if np.all(profile == 0):
            # Cold user fallback
            empty_df = pd.DataFrame(columns=["item_id", "headline", "topic", "score", "keywords"])
            return empty_df

        seen_items = set()
        if exclude_seen and not self.ratings_df.empty:
            seen_items = set(self.ratings_df[self.ratings_df["user_id"] == user_id]["item_id"])

        scores: List[Dict[str, Any]] = []
        for _, row in self.items_df.iterrows():
            item_id = row["item_id"]
            if exclude_seen and item_id in seen_items:
                continue
            item_vec = self.pipeline.get_item_vector(item_id)
            # Dot product of L2 normalized vectors is cosine similarity
            sim = float(np.dot(profile, item_vec))
            sim = max(0.0, min(1.0, sim))
            kw = self.pipeline.explain_match(profile, item_vec, top_n=5)
            scores.append({
                "item_id": item_id,
                "headline": row["headline"],
                "topic": row["topic"],
                "score": sim,
                "keywords": _format_keywords(kw)
            })

        res_df = pd.DataFrame(scores).sort_values(by="score", ascending=False).head(top_k).reset_index(drop=True)
        return res_df

    def recommend_for_query(self, query_text: str, top_k: int = 5) -> pd.DataFrame:
        """Recommend articles matching a free-form natural language query."""
        query_vec = self.pipeline.vectorize_query(query_text)
        scores: List[Dict[str, Any]] = []

        for _, row in self.items_df.iterrows():
            item_id = row["item_id"]
            item_vec = self.pipeline.get_item_vector(item_id)
            sim = float(np.dot(query_vec, item_vec))
            sim = max(0.0, min(1.0, sim))
            kw = self.pipeline.explain_match(query_vec, item_vec, top_n=5)
            scores.append({
                "item_id": item_id,
                "headline": row["headline"],
                "topic": row["topic"],
                "score": sim,
                "keywords": _format_keywords(kw)
            })

        res_df = pd.DataFrame(scores).sort_values(by="score", ascending=False).head(top_k).reset_index(drop=True)
        return res_df

    def recommend_similar_items(self, item_id: str, top_k: int = 5) -> pd.DataFrame:
        """Find the top-k most similar items to a given existing item."""
        target_vec = self.pipeline.get_item_vector(item_id)
        scores: List[Dict[str, Any]] = []

        for _, row in self.items_df.iterrows():
            curr_id = row["item_id"]
            if curr_id == item_id:
                continue
            curr_vec = self.pipeline.get_item_vector(curr_id)
            sim = float(np.dot(target_vec, curr_vec))
            sim = max(0.0, min(1.0, sim))
            kw = self.pipeline.explain_match(target_vec, curr_vec, top_n=5)
            scores.append({
                "item_id": curr_id,
                "headline": row["headline"],
                "topic": row["topic"],
                "score": sim,
                "keywords": _format_keywords(kw)
            })

        res_df = pd.DataFrame(scores).sort_values(by="score", ascending=False).head(top_k).reset_index(drop=True)
        return res_df

    def recommend_for_new_article(
        self,
        headline: str,
        summary_text: str,
        tags: str = "",
        top_k: int = 5,
        top_users_k: int = 3
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Cold-start handling: recommend similar existing items AND identify likely readers
        for a brand-new, unrated article.
        
        Returns:
            Tuple of (similar_items_df, top_readers_df)
        """
        # Composite text matching fit_transform formula: headline x2 + summary + tags
        tags_clean = tags.replace(";", " ")
        doc_raw = f"{headline} {headline} {summary_text} {tags_clean}"
        article_vec = self.pipeline.vectorize_query(doc_raw)

        # 1. Similar existing items
        similar_items: List[Dict[str, Any]] = []
        for _, row in self.items_df.iterrows():
            item_id = row["item_id"]
            item_vec = self.pipeline.get_item_vector(item_id)
            sim = float(np.dot(article_vec, item_vec))
            sim = max(0.0, min(1.0, sim))
            kw = self.pipeline.explain_match(article_vec, item_vec, top_n=5)
            similar_items.append({
                "item_id": item_id,
                "headline": row["headline"],
                "topic": row["topic"],
                "score": sim,
                "keywords": _format_keywords(kw)
            })
        sim_df = pd.DataFrame(similar_items).sort_values(by="score", ascending=False).head(top_k).reset_index(drop=True)

        # 2. Top likely readers by user profile cosine
        user_scores: List[Dict[str, Any]] = []
        for uid, profile in self.user_profiles.items():
            if np.all(profile == 0):
                continue
            sim = float(np.dot(profile, article_vec))
            sim = max(0.0, min(1.0, sim))
            kw = self.pipeline.explain_match(profile, article_vec, top_n=5)
            user_scores.append({
                "user_id": uid,
                "score": sim,
                "keywords": _format_keywords(kw)
            })
        readers_df = pd.DataFrame(user_scores).sort_values(by="score", ascending=False).head(top_users_k).reset_index(drop=True)

        return sim_df, readers_df


class UserBasedCF:
    """User-Based Collaborative Filtering using mean-centered ratings and cosine similarity.
    Predicts unseen items using top-k nearest neighbors.
    """

    def __init__(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame, k_neighbors: int = 5):
        self.items_df = items_df.copy().reset_index(drop=True)
        self.ratings_df = ratings_df.copy().reset_index(drop=True)
        self.k_neighbors = k_neighbors
        self.item_metadata = dict(zip(self.items_df["item_id"], zip(self.items_df["headline"], self.items_df["topic"], self.items_df["tags"])))
        
        # Build user-item pivot table
        if not self.ratings_df.empty:
            self.user_item_matrix = self.ratings_df.pivot(index="user_id", columns="item_id", values="rating")
            self.user_means = self.user_item_matrix.mean(axis=1)
            self.global_mean = float(self.ratings_df["rating"].mean())
            # Mean-centered matrix for cosine similarity
            self.centered_matrix = self.user_item_matrix.sub(self.user_means, axis=0).fillna(0.0)
            
            # Compute user-user similarity matrix
            users = self.user_item_matrix.index.tolist()
            user_sim_arr = cosine_similarity(self.centered_matrix.values)
            self.user_sim_df = pd.DataFrame(user_sim_arr, index=users, columns=users)
        else:
            self.user_item_matrix = pd.DataFrame()
            self.user_means = pd.Series(dtype=float)
            self.global_mean = 3.0
            self.centered_matrix = pd.DataFrame()
            self.user_sim_df = pd.DataFrame()

    def predict_rating(self, user_id: str, item_id: str) -> float:
        """Predict rating for user_id on item_id using top-k neighbors."""
        if user_id not in self.user_means:
            return self.global_mean

        user_mean = float(self.user_means[user_id])
        
        if self.user_item_matrix.empty or item_id not in self.user_item_matrix.columns:
            return user_mean

        # Find users who have rated this item
        item_ratings = self.user_item_matrix[item_id].dropna()
        if item_ratings.empty:
            return user_mean

        # Other users who rated this item
        other_raters = [u for u in item_ratings.index if u != user_id]
        if not other_raters:
            return user_mean

        # Similarities with these raters
        sims = self.user_sim_df.loc[user_id, other_raters]
        
        # Filter for positive similarity only
        pos_sims = sims[sims > 0]
        if pos_sims.empty:
            return user_mean

        # Select top-k neighbors
        top_neighbors = pos_sims.sort_values(ascending=False).head(self.k_neighbors)
        
        # Weighted sum of mean-centered ratings
        weighted_sum = 0.0
        sim_sum = 0.0
        for neighbor, sim_weight in top_neighbors.items():
            neighbor_rating = self.user_item_matrix.loc[neighbor, item_id]
            neighbor_mean = self.user_means[neighbor]
            weighted_sum += sim_weight * (neighbor_rating - neighbor_mean)
            sim_sum += abs(sim_weight)

        if sim_sum == 0:
            return user_mean

        pred = user_mean + (weighted_sum / sim_sum)
        # Clip to valid rating range [1.0, 5.0]
        return float(np.clip(pred, 1.0, 5.0))

    def recommend_for_user(self, user_id: str, top_k: int = 5, exclude_seen: bool = True) -> pd.DataFrame:
        """Recommend top-k unseen items for user_id sorted by predicted rating."""
        if user_id not in self.user_means:
            return pd.DataFrame(columns=["item_id", "headline", "topic", "score", "keywords"])

        seen_items = set()
        if exclude_seen and not self.ratings_df.empty:
            seen_items = set(self.ratings_df[self.ratings_df["user_id"] == user_id]["item_id"])

        predictions: List[Dict[str, Any]] = []
        for _, row in self.items_df.iterrows():
            item_id = row["item_id"]
            if exclude_seen and item_id in seen_items:
                continue
            pred_score = self.predict_rating(user_id, item_id)
            tags_list = str(row.get("tags", "")).split(";")
            kw = ", ".join(tags_list[:3]) if tags_list else "cf-recommendation"
            predictions.append({
                "item_id": item_id,
                "headline": row["headline"],
                "topic": row["topic"],
                "score": pred_score,
                "keywords": kw
            })

        res_df = pd.DataFrame(predictions).sort_values(by="score", ascending=False).head(top_k).reset_index(drop=True)
        return res_df


class HybridRecommender:
    """Hybrid Recommender combining Content-Based cosine similarity and User-Based CF predicted ratings.
    Formula: score = alpha * minmax(content_score) + (1 - alpha) * minmax(cf_score)
    Falls back to content-based recommendation when CF has no collaborative signal.
    """

    def __init__(
        self,
        content_recommender: ContentBasedRecommender,
        cf_recommender: UserBasedCF,
        alpha: float = 0.5
    ):
        self.content_rec = content_recommender
        self.cf_rec = cf_recommender
        self.alpha = float(alpha)
        self.items_df = content_recommender.items_df

    def recommend_for_user(self, user_id: str, top_k: int = 5, exclude_seen: bool = True) -> pd.DataFrame:
        """Generate hybrid recommendations for user_id."""
        seen_items = set()
        if exclude_seen and not self.content_rec.ratings_df.empty:
            seen_items = set(self.content_rec.ratings_df[self.content_rec.ratings_df["user_id"] == user_id]["item_id"])

        # Gather unread candidate items
        candidate_items = [row for _, row in self.items_df.iterrows() if not (exclude_seen and row["item_id"] in seen_items)]
        if not candidate_items:
            return pd.DataFrame(columns=["item_id", "headline", "topic", "score", "keywords"])

        # Compute raw content scores and raw CF scores
        content_scores: Dict[str, float] = {}
        content_keywords: Dict[str, str] = {}
        user_profile = self.content_rec.user_profiles.get(user_id, None)
        
        has_content_signal = user_profile is not None and not np.all(user_profile == 0)
        for item in candidate_items:
            item_id = item["item_id"]
            if has_content_signal:
                item_vec = self.content_rec.pipeline.get_item_vector(item_id)
                sim = float(np.dot(user_profile, item_vec))
                kw = self.content_rec.pipeline.explain_match(user_profile, item_vec, top_n=5)
                content_scores[item_id] = max(0.0, min(1.0, sim))
                content_keywords[item_id] = _format_keywords(kw)
            else:
                content_scores[item_id] = 0.0
                content_keywords[item_id] = "general"

        cf_scores: Dict[str, float] = {}
        for item in candidate_items:
            item_id = item["item_id"]
            cf_scores[item_id] = self.cf_rec.predict_rating(user_id, item_id)

        # Min-max normalization for combination
        c_vals = list(content_scores.values())
        c_min, c_max = min(c_vals), max(c_vals)
        c_range = c_max - c_min if c_max > c_min else 1.0

        cf_vals = list(cf_scores.values())
        cf_min, cf_max = min(cf_vals), max(cf_vals)
        cf_range = cf_max - cf_min if cf_max > cf_min else 1.0

        # Check if CF provides differentiation
        has_cf_signal = cf_max > cf_min

        results: List[Dict[str, Any]] = []
        for item in candidate_items:
            item_id = item["item_id"]
            norm_content = (content_scores[item_id] - c_min) / c_range
            norm_cf = (cf_scores[item_id] - cf_min) / cf_range

            if not has_cf_signal:
                final_score = norm_content
            elif not has_content_signal:
                final_score = norm_cf
            else:
                final_score = (self.alpha * norm_content) + ((1.0 - self.alpha) * norm_cf)

            results.append({
                "item_id": item_id,
                "headline": item["headline"],
                "topic": item["topic"],
                "score": float(final_score),
                "keywords": content_keywords[item_id]
            })

        res_df = pd.DataFrame(results).sort_values(by="score", ascending=False).head(top_k).reset_index(drop=True)
        return res_df
