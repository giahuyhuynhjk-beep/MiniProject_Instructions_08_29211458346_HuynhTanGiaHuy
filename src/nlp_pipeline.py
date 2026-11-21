"""Natural Language Processing Pipeline for Tech Article Recommender.
CS467 Advanced AI - Assignment 15 (MP-A08)
Student: Huynh Tan Gia Huy (ID 29211458346)

Contains text cleaning, tokenization, stopword removal (preserving domain tokens like
'linux', 'ai', 'llm'), TF-IDF vectorization, user profile modeling, and explanation.
"""

from __future__ import annotations

import re
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize

# Domain keywords that MUST be preserved even if matched by generic stop-word patterns
PROTECTED_DOMAIN_WORDS = {"linux", "ai", "llm", "cpu", "gpu", "npu", "os", "c", "go"}

# Custom list of domain-irrelevant stopwords to complement scikit-learn stop words
ADDITIONAL_STOPWORDS = {
    "article", "report", "news", "today", "yesterday", "recent", "recently",
    "announced", "announces", "published", "according", "week", "month", "year"
}

# Combine and ensure protected words are excluded from stopword filtering
EFFECTIVE_STOPWORDS = (set(ENGLISH_STOP_WORDS) | ADDITIONAL_STOPWORDS) - PROTECTED_DOMAIN_WORDS


def clean_text(text: str) -> str:
    """Clean text by lowercasing, stripping punctuation, digits, and extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Replace non-alphanumeric characters with spaces (preserving hyphens inside words if needed)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize a cleaned text string into whitespace-separated word tokens."""
    if not text:
        return []
    return text.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove stopwords from token list while preserving essential domain tokens."""
    return [
        token for token in tokens
        if (token in PROTECTED_DOMAIN_WORDS) or (token not in EFFECTIVE_STOPWORDS and len(token) > 1)
    ]


def preprocess(text: str) -> str:
    """End-to-end preprocessing: clean, tokenize, remove stopwords, rejoin into a string."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    filtered = remove_stopwords(tokens)
    return " ".join(filtered)


class NLPPipeline:
    """End-to-end NLP and TF-IDF pipeline for indexing and vectorizing articles."""

    def __init__(self, ngram_range: Tuple[int, int] = (1, 2), sublinear_tf: bool = True):
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.items_df: Optional[pd.DataFrame] = None
        self.item_matrix: Optional[np.ndarray] = None
        self.feature_names: Optional[np.ndarray] = None
        self.item_id_to_idx: Dict[str, int] = {}
        self.idx_to_item_id: Dict[int, str] = {}

    def fit_transform(self, items_df: pd.DataFrame) -> np.ndarray:
        """Construct item documents (headline x2 + summary_text + tags) and fit TF-IDF vectorizer.
        
        Args:
            items_df: DataFrame containing item_id, headline, summary_text, tags.
            
        Returns:
            L2-normalized item TF-IDF matrix of shape (n_items, n_features).
        """
        self.items_df = items_df.copy().reset_index(drop=True)
        
        # Build composite document string with double headline weight
        composite_docs: List[str] = []
        for _, row in self.items_df.iterrows():
            headline = str(row["headline"])
            summary = str(row["summary_text"])
            tags = str(row.get("tags", "")).replace(";", " ")
            # Composite doc: headline x2 + summary_text + tags
            doc_raw = f"{headline} {headline} {summary} {tags}"
            composite_docs.append(preprocess(doc_raw))

        # Initialize TF-IDF vectorizer
        # We supply custom preprocessed docs, so standard word token pattern applies
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            norm="l2"
        )
        
        sparse_matrix = self.vectorizer.fit_transform(composite_docs)
        self.item_matrix = sparse_matrix.toarray()
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        # Map item IDs to index positions
        self.item_id_to_idx = {row["item_id"]: idx for idx, row in self.items_df.iterrows()}
        self.idx_to_item_id = {idx: row["item_id"] for idx, row in self.items_df.iterrows()}
        
        return self.item_matrix

    def vectorize_query(self, text: str) -> np.ndarray:
        """Vectorize an arbitrary text query or new article using the fitted vectorizer.
        
        Returns:
            L2-normalized 1D numpy array of shape (n_features,).
        """
        if self.vectorizer is None:
            raise ValueError("NLPPipeline must be fitted before vectorizing queries.")
        
        preprocessed = preprocess(text)
        vec = self.vectorizer.transform([preprocessed]).toarray()[0]
        # Ensure L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def get_item_vector(self, item_id: str) -> np.ndarray:
        """Retrieve the TF-IDF vector for an existing item ID."""
        if item_id not in self.item_id_to_idx:
            raise KeyError(f"Item ID '{item_id}' not found in pipeline vocabulary.")
        idx = self.item_id_to_idx[item_id]
        return self.item_matrix[idx]

    def build_user_profile(self, user_id: str, ratings_df: pd.DataFrame) -> np.ndarray:
        """Build rating-weighted average TF-IDF vector for a specific user.
        
        Accepts any ratings_df (typically training set only to prevent leakage).
        Formula: sum(rating_i * item_vec_i) / sum(ratings), normalized by L2.
        
        Returns:
            L2-normalized 1D vector of shape (n_features,). Zero vector if no ratings.
        """
        if self.item_matrix is None or self.vectorizer is None:
            raise ValueError("NLPPipeline must be fitted first.")
            
        n_features = self.item_matrix.shape[1]
        user_ratings = ratings_df[ratings_df["user_id"] == user_id]
        
        if user_ratings.empty:
            return np.zeros(n_features)
            
        profile = np.zeros(n_features, dtype=float)
        weight_sum = 0.0
        
        for _, row in user_ratings.iterrows():
            item_id = row["item_id"]
            if item_id in self.item_id_to_idx:
                rating = float(row["rating"])
                item_vec = self.get_item_vector(item_id)
                profile += rating * item_vec
                weight_sum += rating
                
        if weight_sum > 0:
            profile /= weight_sum
            
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile = profile / norm
            
        return profile

    def build_all_user_profiles(self, ratings_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Construct user profile vectors for all users present in the provided ratings DataFrame."""
        user_ids = sorted(ratings_df["user_id"].unique())
        return {uid: self.build_user_profile(uid, ratings_df) for uid in user_ids}

    def explain_match(self, vec_a: np.ndarray, vec_b: np.ndarray, top_n: int = 5) -> List[str]:
        """Explain the match between two vectors (e.g., query vs item, or user vs item)
        by finding terms with the highest elementwise product contribution.
        
        Returns:
            List of top_n keyword strings.
        """
        if self.feature_names is None:
            return []
            
        contributions = vec_a * vec_b
        if np.all(contributions <= 0):
            # Fallback to strongest terms in vec_b if no direct positive overlap
            top_indices = np.argsort(vec_b)[::-1][:top_n]
            return [self.feature_names[i] for i in top_indices if vec_b[i] > 0]
            
        top_indices = np.argsort(contributions)[::-1][:top_n]
        return [self.feature_names[i] for i in top_indices if contributions[i] > 0]

    def get_top_user_keywords(self, user_profile: np.ndarray, top_n: int = 8) -> List[Tuple[str, float]]:
        """Return the highest-weighted terms in a user's profile vector."""
        if self.feature_names is None or np.all(user_profile == 0):
            return []
        top_indices = np.argsort(user_profile)[::-1][:top_n]
        return [(self.feature_names[i], round(float(user_profile[i]), 4)) for i in top_indices if user_profile[i] > 0]
