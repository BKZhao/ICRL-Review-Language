#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import io
import json
import math
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from matplotlib.colors import to_hex, to_rgb
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
ARCHIVE_PATH = PROJECT_ROOT / "DOC" / "archive (1)(1).zip"
LEGACY_RQ1_PATH = PROJECT_ROOT / "DOC" / "8011_rq1.zip"
LEGACY_RQ3_PATH = PROJECT_ROOT / "DOC" / "rq1+3.zip"
DATA_DIR = ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"
FIG_DIR = ROOT / "figures"
APPENDIX_PATH = ROOT / "appendix_tables.tex"
NUMBERS_PATH = DERIVED_DIR / "numbers.tex"
YEARS = list(range(2018, 2024))
RNG = np.random.default_rng(42)

TOKEN_RE = re.compile(r"[a-z']+")
NUMBER_RE = re.compile(r"(-?\d+(?:\.\d+)?)")

POLITE_POSITIVE = {
    "please",
    "thank",
    "thanks",
    "appreciate",
    "appreciated",
    "helpful",
    "interesting",
    "clear",
    "clearly",
    "valuable",
    "strong",
    "nicely",
}
POLITE_NEGATIVE = {
    "unclear",
    "weak",
    "poor",
    "flawed",
    "inadequate",
    "confusing",
    "incorrect",
    "unconvincing",
}
TOXIC_MARKERS = {
    "nonsense",
    "garbage",
    "ridiculous",
    "stupid",
    "absurd",
    "terrible",
    "horrible",
    "worthless",
    "idiotic",
}
CONSTRUCTIVE_MARKERS = {
    "should",
    "could",
    "would",
    "suggest",
    "recommend",
    "clarify",
    "explain",
    "include",
    "compare",
    "ablation",
    "experiment",
    "analysis",
    "future",
    "discuss",
    "revise",
    "improve",
    "additional",
}

PALETTE = {
    "ink": "#173042",
    "muted": "#647482",
    "accept": "#17766f",
    "reject": "#c45b2d",
    "accent": "#2e5fd0",
    "accent_soft": "#8ea8e6",
    "gold": "#b8860b",
    "paper": "#ffffff",
    "card": "#ffffff",
    "grid": "#dbe2ea",
    "shade": "#eef4fb",
    "band": "#e4eefc",
    "before": "#9aa7bb",
    "after": "#17766f",
}

PAPER_CONTROLS = [
    "mean_score",
    "score_std",
    "confidence_mean_imputed",
    "confidence_missing",
    "num_reviews",
    "review_length_mean",
    "title_length",
    "abstract_length",
    "keyword_count",
]
PAPER_TONE_VARS = [
    "mean_sentiment",
    "mean_politeness",
    "mean_toxicity",
    "mean_constructiveness",
]
REVIEW_CONTROLS = [
    "confidence_numeric_imputed",
    "confidence_missing",
    "review_length",
    "title_length",
    "abstract_length",
    "keyword_count",
]
REVIEW_TONE_VARS = [
    "sentiment",
    "politeness",
    "toxicity",
    "constructiveness",
]
BALANCE_CORE_VARS = [
    "mean_score",
    "score_std",
    "confidence_mean_imputed",
    "num_reviews",
    "review_length_mean",
    "title_length",
    "abstract_length",
    "keyword_count",
]


def ensure_dirs() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def blend_colors(color_a: str, color_b: str, weight: float) -> str:
    weight = float(np.clip(weight, 0.0, 1.0))
    rgb_a = np.asarray(to_rgb(color_a))
    rgb_b = np.asarray(to_rgb(color_b))
    blended = (1.0 - weight) * rgb_a + weight * rgb_b
    return to_hex(blended)


def score_bin_colors(bin_orders: pd.Series | np.ndarray) -> list[str]:
    colors = []
    for value in np.asarray(bin_orders, dtype=int):
        if value <= 2:
            colors.append(PALETTE["reject"])
        elif value <= 6:
            colors.append(PALETTE["accent_soft"])
        else:
            colors.append(PALETTE["accept"])
    return colors


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=PALETTE["ink"],
    )


def read_csv_from_zip(zip_path: Path, member: str) -> pd.DataFrame:
    if not zip_path.exists():
        return pd.DataFrame()
    with ZipFile(zip_path) as archive:
        if member not in archive.namelist():
            return pd.DataFrame()
        return pd.read_csv(archive.open(member))


def read_png_from_zip(zip_path: Path, member: str):
    if not zip_path.exists():
        return None
    with ZipFile(zip_path) as archive:
        if member not in archive.namelist():
            return None
        return plt.imread(io.BytesIO(archive.read(member)), format="png")


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value)
    text = text.replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def first_nonempty(values: Iterable[str]) -> str:
    for value in values:
        cleaned = normalize_text(value)
        if cleaned:
            return cleaned
    return ""


def parse_numeric_prefix(value: object) -> float:
    text = normalize_text(value)
    if not text:
        return np.nan
    match = NUMBER_RE.search(text)
    return float(match.group(1)) if match else np.nan


def parse_accept(decision_values: Iterable[str]) -> float:
    texts = [normalize_text(value).lower() for value in decision_values if normalize_text(value)]
    if not texts:
        return np.nan
    joined = " || ".join(texts)
    if "invite to workshop" in joined:
        return 0.0
    accept_markers = ["accept", "poster", "spotlight", "oral", "talk", "notable-top-25", "notable-top-5"]
    reject_markers = ["reject", "no judgement", "concerns raised", "withdraw"]
    if any(marker in joined for marker in accept_markers):
        return 1.0
    if any(marker in joined for marker in reject_markers):
        return 0.0
    return np.nan


def parse_keywords(raw_keywords: object) -> list[str]:
    text = normalize_text(raw_keywords)
    if not text:
        return []
    parsed = None
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = None
    items: list[str]
    if isinstance(parsed, (list, tuple)):
        items = [normalize_text(item) for item in parsed if normalize_text(item)]
    else:
        items = [normalize_text(part) for part in re.split(r"[;,|\n]+", text) if normalize_text(part)]
    cleaned = []
    for item in items:
        phrase = " ".join(tokenize(item))
        if phrase:
            cleaned.append(phrase)
    return cleaned


def keyword_document(keywords: list[str]) -> str:
    return " ; ".join(keywords) if keywords else "missingkeyword"


def language_features(text: str, analyzer: SentimentIntensityAnalyzer) -> dict[str, float]:
    tokens = tokenize(text)
    n_tokens = max(len(tokens), 1)
    counts = Counter(tokens)
    politeness = 100.0 * (
        sum(counts[word] for word in POLITE_POSITIVE) - sum(counts[word] for word in POLITE_NEGATIVE)
    ) / n_tokens
    toxicity = 100.0 * sum(counts[word] for word in TOXIC_MARKERS) / n_tokens
    constructiveness = 100.0 * sum(counts[word] for word in CONSTRUCTIVE_MARKERS) / n_tokens
    sentiment = analyzer.polarity_scores(text)["compound"] if text else 0.0
    return {
        "review_length": float(n_tokens),
        "sentiment": float(sentiment),
        "politeness": float(politeness),
        "toxicity": float(toxicity),
        "constructiveness": float(constructiveness),
    }


def review_text_parts_for_year(year: int, type_map: dict[str, list[str]]) -> list[list[str]]:
    if year <= 2021:
        return [type_map.get("review", [])]
    if year == 2022:
        return [type_map.get("summary_of_the_paper", []), type_map.get("summary_of_the_review", [])]
    return [
        type_map.get("summary_of_the_paper", []),
        type_map.get("strength_and_weaknesses", []),
        type_map.get("summary_of_the_review", []),
    ]


def review_text_source_label(year: int) -> str:
    if year <= 2021:
        return "review"
    if year == 2022:
        return "summary_of_the_paper + summary_of_the_review"
    return "summary_of_the_paper + strength_and_weaknesses + summary_of_the_review"


def join_review_parts(parts: Iterable[str]) -> str:
    cleaned = [normalize_text(part) for part in parts if normalize_text(part)]
    return "\n\n".join(cleaned)


def load_archive_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analyzer = SentimentIntensityAnalyzer()
    review_rows: list[dict[str, object]] = []
    paper_meta_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    with ZipFile(ARCHIVE_PATH) as archive:
        for year in YEARS:
            papers = pd.read_csv(archive.open(f"iclr_{year}_papers.csv"))
            links = pd.read_csv(archive.open(f"iclr_{year}_links.csv"))
            reviews = pd.read_csv(archive.open(f"iclr_{year}_reviews.csv")).fillna("")
            papers = papers.merge(links, on="ID", how="left")
            grouped = {forum: frame for forum, frame in reviews.groupby("Forum", sort=False)}

            for row in papers.itertuples(index=False):
                forum = row.Forum
                forum_reviews = grouped.get(forum)
                type_map: dict[str, list[str]] = defaultdict(list)
                if forum_reviews is not None:
                    for review_row in forum_reviews.itertuples(index=False):
                        type_map[str(review_row.Type)].append(normalize_text(review_row.Content))

                title = normalize_text(getattr(row, "Title", ""))
                abstract = first_nonempty(type_map.get("abstract", []))
                keywords = parse_keywords(first_nonempty(type_map.get("keywords", [])))
                decision_source = "recommendation" if year == 2019 else "decision"
                decision_values = [value for value in type_map.get(decision_source, []) if normalize_text(value)]
                decision_raw = decision_values[-1] if decision_values else ""
                accept = parse_accept(decision_values)
                decision_parse_success = int(not pd.isna(accept))
                text_source = review_text_source_label(year)

                score_type = "rating" if year <= 2021 else "recommendation"
                score_texts = type_map.get(score_type, [])
                if year == 2019:
                    confidence_texts = [
                        value for value in type_map.get("confidence", []) if "area chair" not in value.lower()
                    ]
                else:
                    confidence_texts = type_map.get("confidence", [])
                text_lists = review_text_parts_for_year(year, type_map)
                n_reviews = len(score_texts)
                review_text_count = max((len(items) for items in text_lists), default=0)

                paper_meta_rows.append(
                    {
                        "forum": forum,
                        "year": year,
                        "paper_id": int(row.ID),
                        "title": title,
                        "abstract": abstract,
                        "keywords_raw": first_nonempty(type_map.get("keywords", [])),
                        "keyword_list": keywords,
                        "keyword_document": keyword_document(keywords),
                        "keyword_count": len(keywords),
                        "title_length": len(tokenize(title)),
                        "abstract_length": len(tokenize(abstract)),
                        "decision_raw": decision_raw,
                        "decision_parse_success": decision_parse_success,
                        "accept": accept,
                        "link": normalize_text(getattr(row, "Link", "")),
                    }
                )

                audit_rows.append(
                    {
                        "year": year,
                        "forum": forum,
                        "parsed_reviews": n_reviews,
                        "score_count": len(score_texts),
                        "confidence_count": len(confidence_texts),
                        "review_text_count": review_text_count,
                        "has_decision": int(not pd.isna(accept)),
                    }
                )

                for review_index in range(n_reviews):
                    parts = []
                    for text_list in text_lists:
                        if review_index < len(text_list):
                            parts.append(text_list[review_index])
                    review_text = join_review_parts(parts)
                    features = language_features(review_text, analyzer)
                    review_rows.append(
                        {
                            "forum": forum,
                            "review_id": f"{forum}_{review_index + 1}",
                            "review_index": review_index + 1,
                            "year": year,
                            "review_text": review_text,
                            "review_text_source": text_source,
                            "rating_raw": score_texts[review_index] if review_index < len(score_texts) else "",
                            "rating_numeric": parse_numeric_prefix(
                                score_texts[review_index] if review_index < len(score_texts) else ""
                            ),
                            "rating_parse_success": int(
                                not pd.isna(
                                    parse_numeric_prefix(
                                        score_texts[review_index] if review_index < len(score_texts) else ""
                                    )
                                )
                            ),
                            "confidence_raw": (
                                confidence_texts[review_index] if review_index < len(confidence_texts) else ""
                            ),
                            "confidence_numeric": parse_numeric_prefix(
                                confidence_texts[review_index] if review_index < len(confidence_texts) else ""
                            ),
                            "confidence_parse_success": int(
                                not pd.isna(
                                    parse_numeric_prefix(
                                        confidence_texts[review_index] if review_index < len(confidence_texts) else ""
                                    )
                                )
                            ),
                            "decision_raw": decision_raw,
                            "decision_parse_success": decision_parse_success,
                            "accept": accept,
                            "title": title,
                            "abstract": abstract,
                            "title_length": len(tokenize(title)),
                            "abstract_length": len(tokenize(abstract)),
                            "keywords_raw": first_nonempty(type_map.get("keywords", [])),
                            "keyword_count": len(keywords),
                            "keyword_document": keyword_document(keywords),
                            "link": normalize_text(getattr(row, "Link", "")),
                            "parse_status": "",
                            **features,
                        }
                    )

    review_df = pd.DataFrame(review_rows)
    paper_meta_df = pd.DataFrame(paper_meta_rows)
    audit_df = pd.DataFrame(audit_rows)

    paper_meta_df["accept"] = pd.to_numeric(paper_meta_df["accept"], errors="coerce")
    review_df["accept"] = pd.to_numeric(review_df["accept"], errors="coerce")
    review_df = review_df[review_df["accept"].notna()].copy()
    paper_meta_df = paper_meta_df[paper_meta_df["accept"].notna()].copy()

    return review_df, paper_meta_df, audit_df


def assign_topic_clusters(paper_meta_df: pd.DataFrame, n_clusters: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    docs = paper_meta_df["keyword_document"].fillna("missingkeyword")
    vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(docs)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(matrix) + 1
    paper_meta_df = paper_meta_df.copy()
    paper_meta_df["topic_cluster"] = clusters.astype(int)

    terms = np.array(vectorizer.get_feature_names_out())
    topic_rows = []
    for cluster_id in range(1, n_clusters + 1):
        center = kmeans.cluster_centers_[cluster_id - 1]
        top_terms = terms[np.argsort(center)[-5:][::-1]].tolist()
        n_papers = int((paper_meta_df["topic_cluster"] == cluster_id).sum())
        topic_rows.append(
            {
                "topic_cluster": cluster_id,
                "n_papers": n_papers,
                "top_terms": ", ".join(top_terms),
            }
        )
    topic_df = pd.DataFrame(topic_rows)
    return paper_meta_df, topic_df


def build_canonical_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    review_df, paper_meta_df, audit_df = load_archive_tables()
    paper_meta_df, topic_df = assign_topic_clusters(paper_meta_df)
    review_df = review_df.merge(
        paper_meta_df[["forum", "topic_cluster", "keyword_document", "link"]],
        on=["forum", "keyword_document", "link"],
        how="left",
    )

    grouped = review_df.groupby(["forum", "year"], as_index=False)
    paper_level = grouped.agg(
        accept=("accept", "first"),
        decision_raw=("decision_raw", "first"),
        decision_parse_success=("decision_parse_success", "first"),
        title=("title", "first"),
        abstract=("abstract", "first"),
        keywords_raw=("keywords_raw", "first"),
        keyword_document=("keyword_document", "first"),
        keyword_count=("keyword_count", "first"),
        title_length=("title_length", "first"),
        abstract_length=("abstract_length", "first"),
        topic_cluster=("topic_cluster", "first"),
        link=("link", "first"),
        num_reviews=("review_id", "count"),
        mean_score=("rating_numeric", "mean"),
        score_std=("rating_numeric", "std"),
        rating_parse_success_share=("rating_parse_success", "mean"),
        confidence_mean=("confidence_numeric", "mean"),
        confidence_parse_success_share=("confidence_parse_success", "mean"),
        mean_sentiment=("sentiment", "mean"),
        mean_politeness=("politeness", "mean"),
        mean_toxicity=("toxicity", "mean"),
        mean_constructiveness=("constructiveness", "mean"),
        review_length_mean=("review_length", "mean"),
    )
    paper_level["score_std"] = paper_level["score_std"].fillna(0.0)
    paper_level["accept"] = paper_level["accept"].astype(int)
    paper_level["confidence_missing"] = paper_level["confidence_mean"].isna().astype(int)
    confidence_median = float(paper_level["confidence_mean"].median())
    paper_level["confidence_mean_imputed"] = paper_level["confidence_mean"].fillna(confidence_median)
    paper_level["mean_score_percentile"] = paper_level.groupby("year")["mean_score"].rank(method="average", pct=True)
    paper_level["year_topic_cluster"] = paper_level["year"].astype(str) + "_" + paper_level["topic_cluster"].astype(str)
    paper_level["parse_status"] = (
        "decision="
        + paper_level["decision_parse_success"].astype(int).astype(str)
        + ";rating_share="
        + paper_level["rating_parse_success_share"].round(3).astype(str)
        + ";confidence_share="
        + paper_level["confidence_parse_success_share"].round(3).astype(str)
    )

    review_df["accept"] = review_df["accept"].astype(int)
    review_df["confidence_missing"] = review_df["confidence_numeric"].isna().astype(int)
    review_conf_median = float(review_df["confidence_numeric"].median())
    review_df["confidence_numeric_imputed"] = review_df["confidence_numeric"].fillna(review_conf_median)
    review_df["recommend_positive"] = (review_df["rating_numeric"] >= 6).astype(int)
    review_df["parse_status"] = (
        "rating="
        + review_df["rating_parse_success"].astype(int).astype(str)
        + ";confidence="
        + review_df["confidence_parse_success"].astype(int).astype(str)
        + ";decision="
        + review_df["decision_parse_success"].astype(int).astype(str)
    )

    review_df = review_df[
        [
            "review_id",
            "forum",
            "review_index",
            "year",
            "review_text",
            "review_text_source",
            "rating_raw",
            "rating_numeric",
            "rating_parse_success",
            "confidence_raw",
            "confidence_numeric",
            "confidence_parse_success",
            "confidence_numeric_imputed",
            "confidence_missing",
            "sentiment",
            "politeness",
            "toxicity",
            "constructiveness",
            "review_length",
            "topic_cluster",
            "title",
            "abstract",
            "title_length",
            "abstract_length",
            "keywords_raw",
            "keyword_count",
            "decision_raw",
            "decision_parse_success",
            "accept",
            "recommend_positive",
            "parse_status",
            "link",
        ]
    ].sort_values(["year", "forum", "review_index"])

    paper_level = paper_level.sort_values(["year", "forum"]).reset_index(drop=True)
    return review_df, paper_level, topic_df, audit_df


def pretty_term(term: str) -> str:
    mapping = {
        "mean_sentiment": "Paper-level mean review sentiment",
        "mean_politeness": "Paper-level mean review politeness",
        "mean_toxicity": "Paper-level mean review toxicity",
        "mean_constructiveness": "Paper-level mean review constructiveness",
        "mean_score": "Mean reviewer score",
        "score_std": "Reviewer score disagreement",
        "confidence_mean_imputed": "Mean reviewer confidence",
        "confidence_missing": "Confidence missingness indicator",
        "num_reviews": "Number of reviews",
        "review_length_mean": "Mean review length (tokens)",
        "title_length": "Title length (tokens)",
        "abstract_length": "Abstract length (tokens)",
        "keyword_count": "Number of keywords",
        "sentiment": "Review-level sentiment",
        "politeness": "Review-level politeness",
        "toxicity": "Review-level toxicity",
        "constructiveness": "Review-level constructiveness",
        "review_length": "Review length (tokens)",
        "confidence_numeric_imputed": "Reviewer confidence",
    }
    if term.startswith("C(year)"):
        year = term.split("[T.")[-1].rstrip("]")
        return f"Year FE: {year}"
    if term.startswith("C(topic_cluster)"):
        cluster = term.split("[T.")[-1].rstrip("]")
        return f"Topic FE: cluster {cluster}"
    return mapping.get(term, term)


def fit_glm(formula: str, data: pd.DataFrame, cluster_groups: pd.Series | np.ndarray | None = None):
    model = sm.GLM.from_formula(formula, data=data, family=sm.families.Binomial())
    fit_kwargs = {"maxiter": 200, "disp": False}
    if cluster_groups is not None and pd.Series(cluster_groups).nunique() > 1:
        fit_kwargs["cov_type"] = "cluster"
        fit_kwargs["cov_kwds"] = {"groups": np.asarray(cluster_groups)}
    else:
        fit_kwargs["cov_type"] = "HC1"
    try:
        result = model.fit(**fit_kwargs)
    except np.linalg.LinAlgError:
        fit_kwargs.pop("cov_kwds", None)
        fit_kwargs["cov_type"] = "HC1"
        try:
            result = model.fit(**fit_kwargs)
        except np.linalg.LinAlgError:
            fit_kwargs.pop("cov_type", None)
            result = model.fit(**fit_kwargs)
    if np.isnan(np.asarray(result.bse)).any():
        result = model.fit(maxiter=200, disp=False)
    return result


def fit_glm_plain(formula: str, data: pd.DataFrame):
    model = sm.GLM.from_formula(formula, data=data, family=sm.families.Binomial())
    return model.fit(maxiter=200, disp=False)


def invlogit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35, 35)
    return 1 / (1 + np.exp(-values))


def safe_mvnorm_draws(mean: np.ndarray, cov: np.ndarray, draws: int = 400) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = (cov + cov.T) / 2
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 0, None)
        root = eigvecs @ np.diag(np.sqrt(eigvals))
        normals = RNG.standard_normal((draws, len(mean)))
        return mean + normals @ root.T
    except np.linalg.LinAlgError:
        return np.repeat(np.asarray(mean, dtype=float)[None, :], draws, axis=0)


def summarise_model(result, focal_terms: list[str], label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = result.params
    conf = result.conf_int()
    coef_df = pd.DataFrame(
        {
            "term": params.index,
            "label": [pretty_term(term) for term in params.index],
            "coef": params.values,
            "std_err": result.bse.values,
            "pvalue": result.pvalues.values,
            "ci_low": conf[0].values,
            "ci_high": conf[1].values,
            "odds_ratio": np.exp(np.clip(params.values, -25, 25)),
            "or_ci_low": np.exp(np.clip(conf[0].values, -25, 25)),
            "or_ci_high": np.exp(np.clip(conf[1].values, -25, 25)),
            "model": label,
            "n": int(result.nobs),
        }
    )

    exog = result.model.exog
    exog_names = result.model.exog_names
    draws = safe_mvnorm_draws(params.to_numpy(), result.cov_params().to_numpy())
    fitted_prob = invlogit(exog @ params.to_numpy())
    local_scale = float(np.mean(fitted_prob * (1 - fitted_prob)))
    ame_rows = []
    for term in focal_terms:
        idx = exog_names.index(term)
        point = float(params.iloc[idx] * local_scale)
        term_draws = draws[:, idx] * local_scale
        ci_low = float(np.percentile(term_draws, 2.5))
        ci_high = float(np.percentile(term_draws, 97.5))
        coef_ci_low = float(conf.loc[term, 0] * local_scale)
        coef_ci_high = float(conf.loc[term, 1] * local_scale)
        coef_width = abs(coef_ci_high - coef_ci_low)
        draw_width = abs(ci_high - ci_low)
        if (
            not np.isfinite(ci_low)
            or not np.isfinite(ci_high)
            or (ci_high - ci_low) < 1e-5
            or not (min(ci_low, ci_high) <= point <= max(ci_low, ci_high))
            or draw_width > max(0.5, 5.0 * max(coef_width, 1e-6))
        ):
            ci_low = coef_ci_low
            ci_high = coef_ci_high
            ame_method = "coef_ci_scaled"
        else:
            ame_method = "draws_scaled"
        ame_rows.append(
            {
                "term": term,
                "label": pretty_term(term),
                "ame": point,
                "ci_low": float(min(ci_low, ci_high)),
                "ci_high": float(max(ci_low, ci_high)),
                "model": label,
                "n": int(result.nobs),
                "ame_scale": local_scale,
                "ame_method": ame_method,
            }
        )
    ame_df = pd.DataFrame(ame_rows)
    return coef_df, ame_df


def build_prediction_grid(result, data: pd.DataFrame, variable: str, grid: np.ndarray, group_label: str = "overall") -> pd.DataFrame:
    design_info = result.model.data.design_info
    params = result.params.to_numpy()
    draws = safe_mvnorm_draws(params, result.cov_params().to_numpy(), draws=300)
    rows = []
    for value in grid:
        scenario = data.copy()
        scenario[variable] = value
        exog = np.asarray(patsy.build_design_matrices([design_info], scenario, return_type="dataframe")[0])
        point = float(invlogit(exog @ params).mean())
        draw_preds = []
        for draw in draws:
            draw_preds.append(float(invlogit(exog @ draw).mean()))
        rows.append(
            {
                variable: float(value),
                "predicted_probability": point,
                "ci_low": float(np.percentile(draw_preds, 2.5)),
                "ci_high": float(np.percentile(draw_preds, 97.5)),
                "group": group_label,
            }
        )
    return pd.DataFrame(rows)


def fit_paper_models(paper_df: pd.DataFrame) -> dict[str, object]:
    paper_formula = (
        "accept ~ mean_sentiment + mean_politeness + mean_toxicity + mean_constructiveness + "
        "mean_score + score_std + confidence_mean_imputed + confidence_missing + num_reviews + "
        "review_length_mean + title_length + abstract_length + keyword_count + C(year) + C(topic_cluster)"
    )
    review_formula = (
        "recommend_positive ~ sentiment + politeness + toxicity + constructiveness + "
        "confidence_numeric_imputed + confidence_missing + review_length + title_length + "
        "abstract_length + keyword_count + C(year) + C(topic_cluster)"
    )
    return {
        "paper_formula": paper_formula,
        "review_formula": review_formula,
    }


def focal_term_ci_is_pathological(result, focal_term: str, limit: float = 25.0) -> bool:
    conf = result.conf_int()
    if focal_term not in conf.index:
        return True
    ci_low, ci_high = conf.loc[focal_term]
    if not np.isfinite(ci_low) or not np.isfinite(ci_high):
        return True
    if abs(ci_high - ci_low) > limit:
        return True
    return max(abs(ci_low), abs(ci_high)) > limit


def subgroup_ame(
    data: pd.DataFrame,
    formula: str,
    group_col: str,
    focal_term: str,
    cluster_col: str | None,
    drop_year_fe: bool = False,
    drop_topic_fe: bool = False,
) -> pd.DataFrame:
    rows = []
    for group_value, group_df in data.groupby(group_col):
        local_formula = formula
        if drop_year_fe:
            local_formula = local_formula.replace(" + C(year)", "")
        if drop_topic_fe:
            local_formula = local_formula.replace(" + C(topic_cluster)", "")
        if group_df["accept"].nunique() < 2:
            continue
        cluster_groups = group_df[cluster_col] if cluster_col else None
        fit_note = "requested_spec"
        try:
            result = fit_glm(local_formula, group_df, cluster_groups=cluster_groups)
            if focal_term_ci_is_pathological(result, focal_term):
                result = fit_glm(local_formula, group_df, cluster_groups=None)
                fit_note = "fallback_hc1_cov"
        except Exception:
            fallback_formula = local_formula.replace(" + C(topic_cluster)", "").replace(" + C(year)", "")
            fit_note = "fallback_without_fixed_effects"
            try:
                result = fit_glm(fallback_formula, group_df, cluster_groups=None)
            except Exception:
                continue
        _, ame = summarise_model(result, [focal_term], f"{group_col}:{group_value}")
        row = ame.iloc[0].to_dict()
        row[group_col] = group_value
        row["n_papers"] = int(len(group_df))
        row["fit_note"] = fit_note
        rows.append(row)
    return pd.DataFrame(rows)


def add_disagreement_groups(paper_df: pd.DataFrame) -> pd.DataFrame:
    out = paper_df.copy()
    out["disagreement_group"] = pd.qcut(
        out["score_std"].rank(method="first"),
        q=3,
        labels=["Low disagreement", "Medium disagreement", "High disagreement"],
    )
    return out


def fit_observational_layers(review_df: pd.DataFrame, paper_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    formulas = fit_paper_models(paper_df)
    paper_result = fit_glm(formulas["paper_formula"], paper_df, cluster_groups=paper_df["year"])
    review_result = fit_glm(formulas["review_formula"], review_df, cluster_groups=review_df["forum"])

    paper_coef, paper_ame = summarise_model(paper_result, PAPER_TONE_VARS + PAPER_CONTROLS, "paper_fe")
    review_coef, review_ame = summarise_model(review_result, REVIEW_TONE_VARS + REVIEW_CONTROLS, "review_fe")

    sentiment_grid = np.linspace(
        float(paper_df["mean_sentiment"].quantile(0.05)),
        float(paper_df["mean_sentiment"].quantile(0.95)),
        60,
    )
    paper_margins = build_prediction_grid(paper_result, paper_df, "mean_sentiment", sentiment_grid)

    paper_with_disagreement = add_disagreement_groups(paper_df)
    heterogeneity_year = subgroup_ame(
        paper_df,
        formulas["paper_formula"],
        group_col="year",
        focal_term="mean_sentiment",
        cluster_col="topic_cluster",
        drop_year_fe=True,
        drop_topic_fe=False,
    )
    heterogeneity_disagreement = subgroup_ame(
        paper_with_disagreement,
        formulas["paper_formula"],
        group_col="disagreement_group",
        focal_term="mean_sentiment",
        cluster_col="year_topic_cluster",
        drop_year_fe=False,
        drop_topic_fe=False,
    )
    heterogeneity_topic = subgroup_ame(
        paper_df,
        formulas["paper_formula"],
        group_col="topic_cluster",
        focal_term="mean_sentiment",
        cluster_col="year",
        drop_year_fe=False,
        drop_topic_fe=True,
    )

    return {
        "paper_coef": paper_coef,
        "paper_ame": paper_ame,
        "paper_margins": paper_margins,
        "review_coef": review_coef,
        "review_ame": review_ame,
        "heterogeneity_year": heterogeneity_year,
        "heterogeneity_disagreement": heterogeneity_disagreement,
        "heterogeneity_topic": heterogeneity_topic,
    }


def standardised_mean_difference(treated: pd.Series, control: pd.Series) -> float:
    treated = treated.astype(float)
    control = control.astype(float)
    pooled = math.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2) if len(treated) > 1 and len(control) > 1 else 0.0
    if pooled == 0 or math.isnan(pooled):
        return 0.0
    return float((treated.mean() - control.mean()) / pooled)


def bootstrap_ci(values: np.ndarray, draws: int = 2000) -> tuple[float, float]:
    if len(values) == 0:
        return (np.nan, np.nan)
    samples = RNG.choice(values, size=(draws, len(values)), replace=True)
    means = samples.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_balance(df: pd.DataFrame, matched_pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []

    before_treated = df[df["high_sentiment"] == 1]
    before_control = df[df["high_sentiment"] == 0]
    matched_treated = None
    matched_control = None
    if matched_pairs is not None and not matched_pairs.empty:
        matched_treated = matched_pairs.rename(columns=lambda col: col.replace("_treated", ""))
        matched_control = matched_pairs.rename(columns=lambda col: col.replace("_control", ""))

    for variable in BALANCE_CORE_VARS:
        row = {
            "variable": variable,
            "label": pretty_term(variable),
            "smd_before": standardised_mean_difference(before_treated[variable], before_control[variable]),
        }
        if matched_treated is not None and matched_control is not None:
            row["smd_after"] = standardised_mean_difference(matched_treated[variable], matched_control[variable])
        else:
            row["smd_after"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def propensity_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(
        df[
            BALANCE_CORE_VARS
            + [
                "topic_cluster",
                "year",
            ]
        ],
        columns=["topic_cluster", "year"],
        drop_first=False,
        dtype=float,
    )


def match_sample(df: pd.DataFrame, treatment_col: str = "high_sentiment") -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    design = propensity_design_matrix(df)
    design = (design - design.mean()) / design.std(ddof=0).replace(0, 1)
    model = LogisticRegression(max_iter=4000, random_state=42)
    model.fit(design, df[treatment_col])
    propensity = model.predict_proba(design)[:, 1]
    propensity = np.clip(propensity, 1e-4, 1 - 1e-4)
    work = df.copy()
    work["propensity_score"] = propensity
    work["propensity_logit"] = np.log(propensity / (1 - propensity))
    caliper = 0.1 * work["propensity_logit"].std(ddof=0)

    matched_rows = []
    for year, year_df in work.groupby("year"):
        treated = year_df[year_df[treatment_col] == 1].sort_values(["propensity_logit", "forum"])
        controls = year_df[year_df[treatment_col] == 0].sort_values(["propensity_logit", "forum"])
        available = controls.index.tolist()
        used_controls: set[int] = set()
        for treated_idx, treated_row in treated.iterrows():
            candidates = controls.loc[[idx for idx in available if idx not in used_controls]].copy()
            if candidates.empty:
                continue
            candidates["distance"] = (candidates["propensity_logit"] - treated_row["propensity_logit"]).abs()
            candidates = candidates[candidates["distance"] <= caliper]
            if candidates.empty:
                continue
            control_idx = candidates.sort_values(["distance", "propensity_logit", "forum"]).index[0]
            used_controls.add(int(control_idx))
            control_row = controls.loc[control_idx]
            pair = {}
            for column in work.columns:
                pair[f"{column}_treated"] = treated_row[column]
                pair[f"{column}_control"] = control_row[column]
            pair["distance"] = float(abs(treated_row["propensity_logit"] - control_row["propensity_logit"]))
            pair["year"] = int(year)
            matched_rows.append(pair)

    matched_pairs = pd.DataFrame(matched_rows)
    if matched_pairs.empty:
        diagnostics = {
            "n_analytic": int(len(work)),
            "n_treated": int((work[treatment_col] == 1).sum()),
            "n_control": int((work[treatment_col] == 0).sum()),
            "n_matched_pairs": 0,
            "caliper": float(caliper),
            "att": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "balance_pass": False,
        }
        return work, matched_pairs, diagnostics

    diffs = matched_pairs["accept_treated"] - matched_pairs["accept_control"]
    att = float(diffs.mean())
    ci_low, ci_high = bootstrap_ci(diffs.to_numpy())
    balance = compute_balance(work, matched_pairs)
    balance_pass = bool((balance["smd_after"].abs() < 0.10).all())
    overlap_lower = max(work.loc[work[treatment_col] == 1, "propensity_score"].min(), work.loc[work[treatment_col] == 0, "propensity_score"].min())
    overlap_upper = min(work.loc[work[treatment_col] == 1, "propensity_score"].max(), work.loc[work[treatment_col] == 0, "propensity_score"].max())
    diagnostics = {
        "n_analytic": int(len(work)),
        "n_treated": int((work[treatment_col] == 1).sum()),
        "n_control": int((work[treatment_col] == 0).sum()),
        "n_matched_pairs": int(len(matched_pairs)),
        "caliper": float(caliper),
        "att": att,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "overlap_lower": float(overlap_lower),
        "overlap_upper": float(overlap_upper),
        "balance_pass": balance_pass,
    }
    return work, matched_pairs, diagnostics


def build_borderline_sample(paper_df: pd.DataFrame, lower: float = 0.35, upper: float = 0.65) -> pd.DataFrame:
    analytic = paper_df.copy()
    analytic["mean_score_percentile"] = analytic.groupby("year")["mean_score"].rank(method="average", pct=True)
    analytic = analytic[(analytic["mean_score_percentile"] >= lower) & (analytic["mean_score_percentile"] <= upper)].copy()
    analytic["mean_sentiment_z"] = analytic.groupby("year")["mean_sentiment"].transform(
        lambda series: (series - series.mean()) / (series.std(ddof=0) if series.std(ddof=0) else 1.0)
    )
    analytic["within_year_sentiment_median"] = analytic.groupby("year")["mean_sentiment_z"].transform("median")
    analytic["high_sentiment"] = (analytic["mean_sentiment_z"] > analytic["within_year_sentiment_median"]).astype(int)
    analytic["borderline_window"] = f"{int(lower * 100)}-{int(upper * 100)}"
    return analytic


def build_tercile_variant(analytic_df: pd.DataFrame) -> pd.DataFrame:
    out = analytic_df.copy()
    out["sentiment_rank"] = out.groupby("year")["mean_sentiment_z"].rank(method="first", pct=True)
    out = out[(out["sentiment_rank"] <= (1 / 3)) | (out["sentiment_rank"] >= (2 / 3))].copy()
    out["high_sentiment"] = (out["sentiment_rank"] >= (2 / 3)).astype(int)
    return out


def run_matching_spec(sample_df: pd.DataFrame, specification: str) -> dict[str, object]:
    work, pairs, diag = match_sample(sample_df)
    balance = compute_balance(work, pairs)
    counts = (
        pairs.groupby("year").size().rename("matched_pairs").reset_index()
        if not pairs.empty
        else pd.DataFrame({"year": [], "matched_pairs": []})
    )
    diag = {**diag, "specification": specification}
    return {
        "sample": sample_df,
        "work": work,
        "pairs": pairs,
        "balance": balance,
        "counts": counts,
        "diag": diag,
    }


def fit_matching_layer(paper_df: pd.DataFrame) -> dict[str, object]:
    primary = run_matching_spec(
        build_borderline_sample(paper_df, lower=0.35, upper=0.65),
        "Primary window 35th-65th percentile",
    )
    narrow = run_matching_spec(
        build_borderline_sample(paper_df, lower=0.40, upper=0.60),
        "Window sensitivity 40th-60th percentile",
    )
    wide = run_matching_spec(
        build_borderline_sample(paper_df, lower=0.30, upper=0.70),
        "Window sensitivity 30th-70th percentile",
    )
    robust = run_matching_spec(
        build_tercile_variant(primary["sample"]),
        "Treatment sensitivity: top vs bottom sentiment tercile",
    )

    warning_path = DERIVED_DIR / "psm_warning.json"
    warning_payload = {
        "primary_balance_pass": bool(primary["diag"]["balance_pass"]),
        "narrow_balance_pass": bool(narrow["diag"]["balance_pass"]),
        "wide_balance_pass": bool(wide["diag"]["balance_pass"]),
        "robust_balance_pass": bool(robust["diag"]["balance_pass"]),
    }
    if not all(warning_payload.values()):
        warning_payload["message"] = (
            "At least one matched design failed the |SMD| < 0.10 balance threshold for the core covariates."
        )
    else:
        warning_payload["message"] = "All matched designs satisfied the |SMD| < 0.10 balance threshold."
    warning_path.write_text(json.dumps(warning_payload, indent=2))

    specs = [primary, narrow, wide, robust]
    effect_rows = pd.DataFrame(
        [
            {
                "specification": spec["diag"]["specification"],
                "att": spec["diag"]["att"],
                "ci_low": spec["diag"]["ci_low"],
                "ci_high": spec["diag"]["ci_high"],
                "matched_pairs": spec["diag"]["n_matched_pairs"],
                "balance_pass": spec["diag"]["balance_pass"],
            }
            for spec in specs
        ]
    )

    overlap_rows = pd.DataFrame(
        [
            {
                "specification": spec["diag"]["specification"],
                **{k: v for k, v in spec["diag"].items() if k != "specification"},
            }
            for spec in specs
        ]
    )

    return {
        "borderline_sample": primary["sample"],
        "primary_work": primary["work"],
        "primary_pairs": primary["pairs"],
        "primary_balance": primary["balance"],
        "primary_counts": primary["counts"],
        "primary_diag": primary["diag"],
        "narrow_work": narrow["work"],
        "narrow_pairs": narrow["pairs"],
        "narrow_balance": narrow["balance"],
        "narrow_diag": narrow["diag"],
        "wide_work": wide["work"],
        "wide_pairs": wide["pairs"],
        "wide_balance": wide["balance"],
        "wide_diag": wide["diag"],
        "robust_work": robust["work"],
        "robust_pairs": robust["pairs"],
        "robust_balance": robust["balance"],
        "robust_diag": robust["diag"],
        "specs": specs,
        "effects": effect_rows,
        "overlap": overlap_rows,
    }


def cross_year_prediction_diagnostics(paper_df: pd.DataFrame) -> pd.DataFrame:
    baseline_formula = (
        "accept ~ mean_score + score_std + confidence_mean_imputed + confidence_missing + "
        "num_reviews + review_length_mean + title_length + abstract_length + keyword_count + C(topic_cluster)"
    )
    extended_formula = baseline_formula + " + mean_sentiment + mean_politeness + mean_toxicity + mean_constructiveness"
    rows = []
    for heldout_year in sorted(paper_df["year"].unique()):
        train = paper_df[paper_df["year"] != heldout_year].copy()
        test = paper_df[paper_df["year"] == heldout_year].copy()
        for label, formula in [("baseline", baseline_formula), ("with_language", extended_formula)]:
            result = fit_glm_plain(formula, train)
            preds = np.clip(result.predict(test), 1e-6, 1 - 1e-6)
            rows.append(
                {
                    "heldout_year": int(heldout_year),
                    "model": label,
                    "auc": float(roc_auc_score(test["accept"], preds)),
                    "log_loss": float(log_loss(test["accept"], preds)),
                    "brier": float(brier_score_loss(test["accept"], preds)),
                    "n_test": int(len(test)),
                }
            )
    metrics = pd.DataFrame(rows)
    baseline = metrics[metrics["model"] == "baseline"].rename(
        columns={"auc": "auc_baseline", "log_loss": "log_loss_baseline", "brier": "brier_baseline"}
    )
    extended = metrics[metrics["model"] == "with_language"].rename(
        columns={"auc": "auc_with_language", "log_loss": "log_loss_with_language", "brier": "brier_with_language"}
    )
    merged = baseline.merge(extended, on=["heldout_year", "n_test"])
    merged["delta_auc"] = merged["auc_with_language"] - merged["auc_baseline"]
    merged["delta_log_loss"] = merged["log_loss_with_language"] - merged["log_loss_baseline"]
    merged["delta_brier"] = merged["brier_with_language"] - merged["brier_baseline"]
    return merged.sort_values("heldout_year").reset_index(drop=True)


def compute_year_difference_effects(paper_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = [
        "mean_sentiment",
        "mean_politeness",
        "mean_toxicity",
        "mean_constructiveness",
    ]
    for feature in features:
        for year, year_df in paper_df.groupby("year", sort=True):
            accepted = year_df.loc[year_df["accept"] == 1, feature].dropna()
            rejected = year_df.loc[year_df["accept"] == 0, feature].dropna()
            effect = float(accepted.mean() - rejected.mean())
            accepted_var = float(accepted.var(ddof=1)) if len(accepted) > 1 else 0.0
            rejected_var = float(rejected.var(ddof=1)) if len(rejected) > 1 else 0.0
            std_err = math.sqrt((accepted_var / max(len(accepted), 1)) + (rejected_var / max(len(rejected), 1)))
            rows.append(
                {
                    "feature": feature,
                    "year": int(year),
                    "effect": effect,
                    "std_err": std_err,
                    "ci_low": effect - 1.96 * std_err,
                    "ci_high": effect + 1.96 * std_err,
                    "n_accept": int(len(accepted)),
                    "n_reject": int(len(rejected)),
                }
            )
    return pd.DataFrame(rows)


def compute_measurement_year_summary(paper_df: pd.DataFrame, review_df: pd.DataFrame) -> pd.DataFrame:
    paper_year = paper_df.groupby("year", as_index=False).agg(
        papers=("forum", "count"),
        acceptance_rate=("accept", "mean"),
        keyword_missing_share=("keyword_count", lambda s: float((s == 0).mean())),
        mean_reviews_per_paper=("num_reviews", "mean"),
    )
    review_year = review_df.groupby("year", as_index=False).agg(
        reviews=("review_id", "count"),
        positive_recommend_share=("recommend_positive", "mean"),
        rating_parse_rate=("rating_parse_success", "mean"),
        confidence_parse_rate=("confidence_parse_success", "mean"),
        mean_review_length=("review_length", "mean"),
        median_review_length=("review_length", "median"),
    )
    return paper_year.merge(review_year, on="year", how="left")


def compute_score_bin_bridge(paper_df: pd.DataFrame) -> pd.DataFrame:
    bins = np.linspace(0, 1, 11)
    labels = [f"{i * 10}-{(i + 1) * 10}" for i in range(10)]
    work = paper_df.copy()
    work["score_bin"] = pd.cut(
        work["mean_score_percentile"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    rows = []
    for score_bin, bin_df in work.groupby("score_bin", observed=False):
        if pd.isna(score_bin) or bin_df.empty:
            continue
        accepted = bin_df.loc[bin_df["accept"] == 1, "mean_sentiment"].dropna()
        rejected = bin_df.loc[bin_df["accept"] == 0, "mean_sentiment"].dropna()
        if len(accepted) == 0 or len(rejected) == 0:
            gap = np.nan
            std_err = np.nan
            ci_low = np.nan
            ci_high = np.nan
        else:
            gap = float(accepted.mean() - rejected.mean())
            accepted_var = float(accepted.var(ddof=1)) if len(accepted) > 1 else 0.0
            rejected_var = float(rejected.var(ddof=1)) if len(rejected) > 1 else 0.0
            std_err = math.sqrt((accepted_var / max(len(accepted), 1)) + (rejected_var / max(len(rejected), 1)))
            ci_low = gap - 1.96 * std_err
            ci_high = gap + 1.96 * std_err
        rows.append(
            {
                "score_bin": str(score_bin),
                "bin_order": labels.index(str(score_bin)),
                "n_papers": int(len(bin_df)),
                "n_accept": int((bin_df["accept"] == 1).sum()),
                "n_reject": int((bin_df["accept"] == 0).sum()),
                "acceptance_rate": float(bin_df["accept"].mean()),
                "accept_mean_sentiment": float(accepted.mean()) if len(accepted) else np.nan,
                "reject_mean_sentiment": float(rejected.mean()) if len(rejected) else np.nan,
                "sentiment_gap": gap,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows).sort_values("bin_order").reset_index(drop=True)


def load_legacy_assets() -> dict[str, pd.DataFrame]:
    legacy = {
        "lexicon_summary": read_csv_from_zip(LEGACY_RQ1_PATH, "outputs/rq1/lexicon_summary.csv"),
        "legacy_multivariable_logit": read_csv_from_zip(LEGACY_RQ1_PATH, "outputs/rq1/multivariable_logit.csv"),
        "legacy_descriptive_stats": read_csv_from_zip(LEGACY_RQ1_PATH, "outputs/rq1/descriptive_stats.csv"),
        "legacy_univariate_group_tests": read_csv_from_zip(LEGACY_RQ1_PATH, "outputs/rq1/univariate_group_tests.csv"),
        "legacy_descriptive_by_year": read_csv_from_zip(LEGACY_RQ3_PATH, "outputs/rq3/descriptive_by_year.csv"),
        "legacy_descriptive_by_disagreement": read_csv_from_zip(
            LEGACY_RQ3_PATH, "outputs/rq3/descriptive_by_disagreement.csv"
        ),
        "legacy_stratified_logit_by_year": read_csv_from_zip(
            LEGACY_RQ3_PATH, "outputs/rq3/stratified_logit_by_year.csv"
        ),
        "legacy_cross_year_validation": read_csv_from_zip(LEGACY_RQ3_PATH, "outputs/rq3/cross_year_validation.csv"),
        "legacy_meta_analysis": read_csv_from_zip(LEGACY_RQ3_PATH, "outputs/rq3/meta_analysis_results.csv"),
    }
    return legacy


def load_existing_canonical_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    review_df = pd.read_csv(DERIVED_DIR / "review_level_canonical.csv")
    paper_df = pd.read_csv(DERIVED_DIR / "paper_level_canonical.csv")
    topic_df = pd.read_csv(DERIVED_DIR / "topic_cluster_summary.csv")
    audit_df = pd.read_csv(DERIVED_DIR / "parsing_audit_by_forum.csv")
    return review_df, paper_df, topic_df, audit_df


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.titlesize": 10.6,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 8.2,
            "axes.facecolor": PALETTE["card"],
            "figure.facecolor": PALETTE["paper"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": PALETTE["muted"],
            "axes.labelcolor": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "savefig.facecolor": PALETTE["paper"],
            "savefig.bbox": "tight",
            "grid.color": PALETTE["grid"],
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def plot_figure1_overview(paper_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1])
    ax_counts = fig.add_subplot(gs[:, 0])
    small_axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]

    by_year = (
        paper_df.groupby(["year", "accept"]).size().rename("n").reset_index()
        .pivot(index="year", columns="accept", values="n")
        .fillna(0)
        .rename(columns={0: "Rejected", 1: "Accepted"})
        .reset_index()
    )
    years = by_year["year"].astype(int).tolist()
    x = np.arange(len(years))
    ax_counts.bar(x, by_year["Rejected"], color=PALETTE["reject"], alpha=0.78, label="Rejected")
    ax_counts.bar(
        x,
        by_year["Accepted"],
        bottom=by_year["Rejected"],
        color=PALETTE["accept"],
        alpha=0.88,
        label="Accepted",
    )
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(years)
    ax_counts.set_xlabel("Conference year")
    ax_counts.set_ylabel("Papers")
    ax_counts.set_title("Yearly cohorts and acceptance volume")
    ax_counts.grid(axis="y", alpha=0.35)
    ax_rate = ax_counts.twinx()
    accept_rate = paper_df.groupby("year")["accept"].mean().reindex(years).to_numpy()
    ax_rate.plot(x, accept_rate, color=PALETTE["accent"], marker="o", linewidth=1.8)
    ax_rate.set_ylabel("Acceptance rate")
    ax_counts.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 1.02), borderaxespad=0.0)

    feature_specs = [
        ("mean_sentiment", "Sentiment"),
        ("mean_politeness", "Politeness"),
        ("mean_toxicity", "Toxicity"),
        ("mean_constructiveness", "Constructiveness"),
    ]
    for ax, (feature, title) in zip(small_axes, feature_specs):
        rejected = paper_df.loc[paper_df["accept"] == 0, feature].dropna().to_numpy()
        accepted = paper_df.loc[paper_df["accept"] == 1, feature].dropna().to_numpy()
        bp = ax.boxplot(
            [rejected, accepted],
            positions=[1, 2],
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": PALETTE["ink"], "linewidth": 1.1},
            whiskerprops={"color": PALETTE["muted"], "linewidth": 0.9},
            capprops={"color": PALETTE["muted"], "linewidth": 0.9},
        )
        for patch, color in zip(bp["boxes"], [PALETTE["reject"], PALETTE["accept"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Reject", "Accept"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    save_figure(fig, "figure1_overview")


def plot_figure1(paper_df: pd.DataFrame, borderline_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), constrained_layout=True)

    bins = np.linspace(0, 1, 22)
    for decision, color, label in [(1, PALETTE["accept"], "Accepted"), (0, PALETTE["reject"], "Rejected")]:
        subset = paper_df.loc[paper_df["accept"] == decision, "mean_score_percentile"]
        axes[0].hist(
            subset,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=label,
        )
    axes[0].axvspan(0.35, 0.65, color=PALETTE["shade"], alpha=0.85)
    axes[0].set_xlabel("Within-year percentile of mean score")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Score overlap motivates the borderline design")
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.50, 1.04), borderaxespad=0.0)
    axes[0].grid(axis="y", alpha=0.4)

    deciles = pd.qcut(
        paper_df["mean_score_percentile"].rank(method="first"),
        10,
        labels=[f"{i * 10}-{(i + 1) * 10}" for i in range(10)],
    )
    rate_df = paper_df.assign(score_decile=deciles).groupby("score_decile", as_index=False).agg(
        accept_rate=("accept", "mean"),
        n=("forum", "count"),
    )
    x = np.arange(len(rate_df))
    axes[1].bar(x, rate_df["n"], color=PALETTE["shade"], edgecolor=PALETTE["grid"])
    axes[1].set_ylabel("Papers per decile")
    axes[1].set_xlabel("Within-year mean-score decile")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(rate_df["score_decile"], rotation=35, ha="right")
    ax2 = axes[1].twinx()
    ax2.plot(x, rate_df["accept_rate"], color=PALETTE["accent"], marker="o", linewidth=1.8)
    ax2.set_ylabel("Acceptance rate")
    middle_idx = [3, 4, 5, 6]
    axes[1].axvspan(min(middle_idx) - 0.5, max(middle_idx) - 0.5, color=PALETTE["shade"], alpha=0.5)
    axes[1].set_title(
        f"Observational layer: full sample\nMatched layer: {len(borderline_df):,} borderline papers"
    )
    axes[1].grid(False)
    ax2.grid(False)

    save_figure(fig, "figure1_design_motivation")


def plot_figure2(paper_margins: pd.DataFrame, paper_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    ax.plot(
        paper_margins["mean_sentiment"],
        paper_margins["predicted_probability"],
        color=PALETTE["accent"],
        linewidth=2.2,
    )
    ax.fill_between(
        paper_margins["mean_sentiment"],
        paper_margins["ci_low"],
        paper_margins["ci_high"],
        color=PALETTE["accent"],
        alpha=0.18,
    )
    ax.set_xlabel("Paper-level mean sentiment")
    ax.set_ylabel("Predicted probability of acceptance")
    ax.set_title("Fixed-effects marginal predicted probability")
    ax.grid(axis="y", alpha=0.4)

    accepted = paper_df.loc[paper_df["accept"] == 1, "mean_sentiment"].sample(
        min(300, int((paper_df["accept"] == 1).sum())),
        random_state=42,
    )
    rejected = paper_df.loc[paper_df["accept"] == 0, "mean_sentiment"].sample(
        min(300, int((paper_df["accept"] == 0).sum())),
        random_state=42,
    )
    ax.scatter(accepted, np.full(len(accepted), ax.get_ylim()[0] + 0.01), s=5, color=PALETTE["accept"], alpha=0.35)
    ax.scatter(rejected, np.full(len(rejected), ax.get_ylim()[0] + 0.03), s=5, color=PALETTE["reject"], alpha=0.25)
    legend_items = [
        Line2D([0], [0], color=PALETTE["accent"], lw=2, label="Model-implied probability"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["accept"], markersize=5, label="Accepted papers"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["reject"], markersize=5, label="Rejected papers"),
    ]
    ax.legend(handles=legend_items, frameon=False, loc="lower right")
    save_figure(fig, "figure2_paper_fe_margins")


def plot_figure3(match_results: dict[str, object]) -> None:
    effects = match_results["effects"]
    balance = match_results["primary_balance"].copy().sort_values("smd_before", key=lambda s: s.abs())

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)

    y = np.arange(len(effects))
    axes[0].axvline(0, color=PALETTE["grid"], linewidth=1.0)
    axes[0].hlines(y, effects["ci_low"], effects["ci_high"], color=PALETTE["ink"], linewidth=1.4)
    point_colors = [PALETTE["gold"]] + [PALETTE["accent"]] * max(len(effects) - 1, 0)
    axes[0].scatter(effects["att"], y, color=point_colors, s=36, zorder=3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(effects["specification"])
    axes[0].set_xlabel("Matched acceptance contrast (ATT estimand)")
    axes[0].set_title("Matched borderline contrast")
    axes[0].grid(axis="x", alpha=0.4)

    y2 = np.arange(len(balance))
    axes[1].axvline(0.10, color=PALETTE["reject"], linestyle="--", linewidth=1.0)
    axes[1].scatter(balance["smd_before"].abs(), y2, color=PALETTE["before"], s=22, label="Before matching")
    axes[1].scatter(balance["smd_after"].abs(), y2, color=PALETTE["after"], s=22, label="After matching")
    axes[1].set_yticks(y2)
    axes[1].set_yticklabels(balance["label"])
    axes[1].set_xlabel("Absolute standardized mean difference")
    axes[1].set_title("Balance in the primary matched sample")
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].grid(axis="x", alpha=0.4)

    save_figure(fig, "figure3_matched_effect_balance")


def plot_figure3_temporal_stability(year_difference_df: pd.DataFrame) -> None:
    plot_df = year_difference_df[year_difference_df["feature"].isin(["mean_sentiment", "mean_politeness"])].copy()
    plot_df["label"] = plot_df["feature"].map(
        {
            "mean_sentiment": "Accepted - rejected sentiment",
            "mean_politeness": "Accepted - rejected politeness",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for ax, feature, title in [
        (axes[0], "mean_sentiment", "Sentiment gap by year"),
        (axes[1], "mean_politeness", "Politeness gap by year"),
    ]:
        data = plot_df[plot_df["feature"] == feature].sort_values("year")
        x = np.arange(len(data))
        ax.axhline(0, color=PALETTE["grid"], linewidth=1.0)
        ax.vlines(x, data["ci_low"], data["ci_high"], color=PALETTE["ink"], linewidth=1.4)
        ax.scatter(x, data["effect"], color=PALETTE["accent"], s=28, zorder=3)
        ax.plot(x, data["effect"], color=PALETTE["accent"], linewidth=1.1, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(data["year"].astype(int))
        ax.set_ylabel("Accepted - rejected difference")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.35)
    save_figure(fig, "figure3_temporal_stability")


def plot_figure4(observational: dict[str, pd.DataFrame]) -> None:
    year_df = observational["heterogeneity_year"].copy().sort_values("year")
    disagreement_df = observational["heterogeneity_disagreement"].copy().sort_values("disagreement_group")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)

    for ax, data, xcol, title in [
        (axes[0], year_df, "year", "Average marginal effect by year"),
        (axes[1], disagreement_df, "disagreement_group", "Average marginal effect by disagreement"),
    ]:
        x = np.arange(len(data))
        ax.axhline(0, color=PALETTE["grid"], linewidth=1.0)
        ax.vlines(x, data["ci_low"], data["ci_high"], color=PALETTE["ink"], linewidth=1.4)
        ax.scatter(x, data["ame"], color=PALETTE["accent"], s=28, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(data[xcol], rotation=0 if xcol == "year" else 18)
        ax.set_ylabel("Average marginal effect of sentiment")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.4)

    save_figure(fig, "figure4_heterogeneity_ame")


def plot_appendix_measurement_summary(
    paper_df: pd.DataFrame, review_df: pd.DataFrame, measurement_year_df: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.35, 5.9),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1], "height_ratios": [1, 1]},
    )

    x = np.arange(len(measurement_year_df))
    years = measurement_year_df["year"].astype(int).astype(str)

    axes[0, 0].bar(
        x,
        measurement_year_df["papers"],
        color=PALETTE["shade"],
        edgecolor=blend_colors(PALETTE["accent"], "#ffffff", 0.55),
        linewidth=1.1,
    )
    ax00_twin = axes[0, 0].twinx()
    ax00_twin.plot(x, measurement_year_df["reviews"], color=PALETTE["accent"], marker="o", linewidth=2.0)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(years)
    axes[0, 0].set_title("Archive coverage by year", loc="left", pad=7, fontweight="bold")
    axes[0, 0].set_ylabel("Papers")
    ax00_twin.set_ylabel("Reviews")
    axes[0, 0].set_ylim(0, 1320)
    axes[0, 0].set_yticks([0, 300, 600, 900, 1200])
    review_min = float(measurement_year_df["reviews"].min())
    review_max = float(measurement_year_df["reviews"].max())
    ax00_twin.set_ylim(review_min - 80, review_max + 220)
    axes[0, 0].grid(axis="y", alpha=0.35)
    ax00_twin.grid(False)
    axes[0, 0].legend(
        handles=[
            Line2D([0], [0], lw=8, color=PALETTE["shade"], label="Papers"),
            Line2D([0], [0], marker="o", lw=2.0, color=PALETTE["accent"], label="Reviews"),
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.03, 0.99),
        borderaxespad=0.0,
    )
    add_panel_label(axes[0, 0], "A")

    axes[0, 1].plot(
        x,
        measurement_year_df["confidence_parse_rate"],
        color=PALETTE["accent"],
        marker="o",
        linewidth=2.0,
        label="Confidence parse rate",
    )
    axes[0, 1].plot(
        x,
        1 - measurement_year_df["keyword_missing_share"],
        color=PALETTE["accept"],
        marker="s",
        linewidth=1.8,
        label="Keyword coverage",
    )
    axes[0, 1].plot(
        x,
        measurement_year_df["positive_recommend_share"],
        color=PALETTE["gold"],
        marker="^",
        linewidth=1.8,
        label="Positive recommendation share",
    )
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(years)
    axes[0, 1].set_ylim(-0.02, 1.08)
    axes[0, 1].set_title("Measurement completeness by year", loc="left", pad=7, fontweight="bold")
    legend = axes[0, 1].legend(
        frameon=True,
        facecolor=PALETTE["paper"],
        edgecolor="none",
        framealpha=0.94,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.02),
        borderaxespad=0.0,
        handletextpad=0.45,
        labelspacing=0.28,
        borderpad=0.25,
    )
    for text in legend.get_texts():
        text.set_fontsize(7.6)
    axes[0, 1].grid(axis="y", alpha=0.35)
    add_panel_label(axes[0, 1], "B")

    review_count_bins = np.arange(0.5, min(int(paper_df["num_reviews"].max()) + 1.5, 12.5), 1)
    axes[1, 0].hist(
        paper_df["num_reviews"],
        bins=review_count_bins,
        color=blend_colors(PALETTE["accept"], "#ffffff", 0.16),
        alpha=0.95,
        edgecolor="#ffffff",
        linewidth=1.0,
    )
    axes[1, 0].set_xticks(sorted(paper_df["num_reviews"].astype(int).unique())[:10])
    axes[1, 0].set_xlabel("Reviews per paper")
    axes[1, 0].set_ylabel("Papers")
    axes[1, 0].set_title("Review-count distribution", loc="left", pad=7, fontweight="bold")
    axes[1, 0].grid(axis="y", alpha=0.35)
    add_panel_label(axes[1, 0], "C")

    accepted_lengths = review_df.loc[review_df["accept"] == 1, "review_length"].dropna().to_numpy()
    rejected_lengths = review_df.loc[review_df["accept"] == 0, "review_length"].dropna().to_numpy()
    bp = axes[1, 1].boxplot(
        [rejected_lengths, accepted_lengths],
        positions=[1, 2],
        widths=0.50,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": PALETTE["ink"], "linewidth": 1.1},
        whiskerprops={"color": PALETTE["muted"], "linewidth": 0.9},
        capprops={"color": PALETTE["muted"], "linewidth": 0.9},
        boxprops={"linewidth": 1.0},
    )
    for patch, color in zip(bp["boxes"], [PALETTE["reject"], PALETTE["accept"]]):
        patch.set_facecolor(blend_colors(color, "#ffffff", 0.45))
        patch.set_edgecolor(color)
    axes[1, 1].set_xticks([1, 2])
    axes[1, 1].set_xticklabels(["Reject", "Accept"])
    axes[1, 1].set_ylabel("Tokens per review")
    axes[1, 1].set_title("Review length by final decision", loc="left", pad=7, fontweight="bold")
    axes[1, 1].grid(axis="y", alpha=0.35)
    add_panel_label(axes[1, 1], "D")

    save_figure(fig, "appendix_figure_measurement_summary")


def plot_appendix_feature_correlations(paper_df: pd.DataFrame) -> None:
    features = [
        "mean_sentiment",
        "mean_politeness",
        "mean_toxicity",
        "mean_constructiveness",
        "mean_score",
        "score_std",
        "confidence_mean_imputed",
        "num_reviews",
        "review_length_mean",
    ]
    corr = paper_df[features].corr().fillna(0.0)
    labels = [
        "Sentiment",
        "Politeness",
        "Toxicity",
        "Constructive",
        "Mean score",
        "Disagreement",
        "Confidence",
        "Reviews",
        "Length",
    ]
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Paper-level feature correlations")
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color=PALETTE["ink"])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    save_figure(fig, "appendix_figure_feature_correlations")


def plot_appendix_score_bridge(bridge_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.35, 3.65),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1]},
    )
    x = np.arange(len(bridge_df))

    zone_colors = score_bin_colors(bridge_df["bin_order"])
    axes[0].axvspan(3.5, 6.5, color=PALETTE["band"], alpha=0.70, zorder=0)
    axes[0].bar(
        x,
        bridge_df["n_papers"],
        color=zone_colors,
        alpha=0.82,
        edgecolor="#ffffff",
        linewidth=1.0,
    )
    ax0_twin = axes[0].twinx()
    ax0_twin.plot(x, bridge_df["acceptance_rate"], color=PALETTE["accent"], marker="o", linewidth=2.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(bridge_df["score_bin"], rotation=35, ha="right")
    axes[0].set_xlabel("Within-year mean-score decile")
    axes[0].set_ylabel("Papers")
    ax0_twin.set_ylabel("Acceptance rate")
    axes[0].set_ylim(0, float(bridge_df["n_papers"].max()) * 1.28)
    ax0_twin.set_ylim(-0.02, 1.10)
    axes[0].set_title("Score bins define the overlap problem", loc="left", pad=7, fontweight="bold")
    axes[0].legend(
        handles=[
            Line2D([0], [0], lw=8, color=blend_colors(PALETTE["accent"], "#ffffff", 0.55), label="Papers per bin"),
            Line2D([0], [0], marker="o", lw=2.0, color=PALETTE["accent"], label="Acceptance rate"),
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.03, 0.99),
        borderaxespad=0.0,
    )
    axes[0].grid(axis="y", alpha=0.35)
    add_panel_label(axes[0], "A")

    clean = bridge_df.dropna(subset=["sentiment_gap", "ci_low", "ci_high"]).copy()
    x1 = np.arange(len(clean))
    axes[1].axhline(0, color=PALETTE["muted"], linewidth=1.0, linestyle="--")
    axes[1].errorbar(
        x1,
        clean["sentiment_gap"],
        yerr=[clean["sentiment_gap"] - clean["ci_low"], clean["ci_high"] - clean["sentiment_gap"]],
        color=PALETTE["accent"],
        marker="o",
        linewidth=1.7,
        markersize=5.2,
        capsize=3,
    )
    axes[1].set_xticks(x1)
    axes[1].set_xticklabels(clean["score_bin"], rotation=35, ha="right")
    axes[1].set_xlabel("Within-year mean-score decile")
    axes[1].set_ylabel("Accepted - rejected sentiment")
    axes[1].set_title("Sentiment gaps within score bins", loc="left", pad=7, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.35)
    add_panel_label(axes[1], "B")

    save_figure(fig, "appendix_figure_score_bridge")


def plot_appendix_psm_overlap(match_results: dict[str, object]) -> None:
    primary_work = match_results["primary_work"]
    primary_pairs = match_results["primary_pairs"]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.35, 3.55),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1]},
    )
    bins = np.linspace(0, 1, 18)
    before_density_max = 1.0
    after_density_max = 1.0

    for value, color, label in [
        (1, PALETTE["accept"], "High sentiment"),
        (0, PALETTE["reject"], "Low sentiment"),
    ]:
        subset = primary_work.loc[primary_work["high_sentiment"] == value, "propensity_score"].dropna()
        if len(subset) > 1:
            hist, _ = np.histogram(subset, bins=bins, density=True)
            before_density_max = max(before_density_max, float(hist.max()))
        axes[0].hist(
            subset,
            bins=bins,
            density=True,
            histtype="stepfilled",
            linewidth=0,
            alpha=0.10,
            color=color,
        )
        axes[0].hist(
            subset,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=label,
        )
    axes[0].set_xlabel("Propensity score")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Borderline sample before matching", loc="left", pad=7, fontweight="bold")
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.03, 0.98), borderaxespad=0.0)
    axes[0].set_ylim(0, before_density_max * 1.14)
    axes[0].grid(axis="y", alpha=0.35)
    add_panel_label(axes[0], "A")

    if not primary_pairs.empty:
        matched_high = primary_pairs["propensity_score_treated"].dropna()
        matched_low = primary_pairs["propensity_score_control"].dropna()
        for subset in [matched_high, matched_low]:
            if len(subset) > 1:
                hist, _ = np.histogram(subset, bins=bins, density=True)
                after_density_max = max(after_density_max, float(hist.max()))
        axes[1].hist(
            matched_high,
            bins=bins,
            density=True,
            histtype="stepfilled",
            linewidth=0,
            alpha=0.10,
            color=PALETTE["accept"],
        )
        axes[1].hist(
            matched_high,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=PALETTE["accept"],
            label="Matched high sentiment",
        )
        axes[1].hist(
            matched_low,
            bins=bins,
            density=True,
            histtype="stepfilled",
            linewidth=0,
            alpha=0.10,
            color=PALETTE["reject"],
        )
        axes[1].hist(
            matched_low,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            color=PALETTE["reject"],
            label="Matched low sentiment",
        )
    axes[1].set_xlabel("Propensity score")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Matched sample after matching", loc="left", pad=7, fontweight="bold")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.03, 0.99), borderaxespad=0.0)
    axes[1].set_ylim(0, after_density_max * 1.24)
    axes[1].grid(axis="y", alpha=0.35)
    add_panel_label(axes[1], "B")

    save_figure(fig, "appendix_figure_psm_overlap")


def plot_appendix_prediction_diagnostics(prediction_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.35, 3.55),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1]},
    )
    plot_df = prediction_df.sort_values("heldout_year").copy()
    x = np.arange(len(plot_df))

    axes[0].plot(x, plot_df["auc_baseline"], color=PALETTE["muted"], marker="o", linewidth=1.7, label="Baseline")
    axes[0].plot(
        x,
        plot_df["auc_with_language"],
        color=PALETTE["accent"],
        marker="o",
        linewidth=1.7,
        label="With language",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["heldout_year"].astype(int))
    axes[0].set_ylabel("Held-out AUC")
    axes[0].set_title("Cross-year predictive performance", loc="left", pad=7, fontweight="bold")
    auc_min = float(min(plot_df["auc_baseline"].min(), plot_df["auc_with_language"].min()))
    auc_max = float(max(plot_df["auc_baseline"].max(), plot_df["auc_with_language"].max()))
    auc_pad = max((auc_max - auc_min) * 0.22, 0.0012)
    axes[0].set_ylim(auc_min - 0.0004, auc_max + auc_pad)
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.03, 0.98), borderaxespad=0.0)
    axes[0].grid(axis="y", alpha=0.35)
    add_panel_label(axes[0], "A")

    width = 0.36
    axes[1].axhline(0, color=PALETTE["muted"], linewidth=1.0, linestyle="--")
    axes[1].bar(x - width / 2, plot_df["delta_auc"], width=width, color=PALETTE["accent"], alpha=0.9, label=r"$\Delta$AUC")
    axes[1].bar(
        x + width / 2,
        plot_df["delta_brier"],
        width=width,
        color=PALETTE["gold"],
        alpha=0.85,
        label=r"$\Delta$Brier",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_df["heldout_year"].astype(int))
    axes[1].set_title("Incremental gains from adding language", loc="left", pad=7, fontweight="bold")
    gain_min = float(min(plot_df["delta_auc"].min(), plot_df["delta_brier"].min()))
    gain_max = float(max(plot_df["delta_auc"].max(), plot_df["delta_brier"].max()))
    gain_pad = max(max(abs(gain_min), abs(gain_max)) * 0.24, 0.00008)
    axes[1].set_ylim(gain_min - gain_pad, gain_max + gain_pad * 2.2)
    axes[1].legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.03, 0.99),
        ncol=1,
        borderaxespad=0.0,
    )
    axes[1].grid(axis="y", alpha=0.35)
    add_panel_label(axes[1], "B")

    save_figure(fig, "appendix_figure_prediction_diagnostics")


def plot_appendix_topic_heterogeneity(observational: dict[str, pd.DataFrame], topic_df: pd.DataFrame) -> None:
    topic_effects = observational["heterogeneity_topic"].copy()
    topic_effects = topic_effects.merge(topic_df[["topic_cluster", "top_terms"]], on="topic_cluster", how="left")
    topic_effects["topic_label"] = topic_effects.apply(
        lambda row: f"{int(row['topic_cluster'])}: {textwrap.shorten(str(row['top_terms']), width=28, placeholder='...')}",
        axis=1,
    )
    topic_effects = topic_effects.sort_values("ame")

    fig, ax = plt.subplots(figsize=(7.35, 5.1), constrained_layout=True)
    y = np.arange(len(topic_effects))
    colors = [PALETTE["accent"] if value >= 0 else PALETTE["reject"] for value in topic_effects["ame"]]
    ax.axvline(0, color=PALETTE["muted"], linewidth=1.0, linestyle="--")
    ax.errorbar(
        topic_effects["ame"],
        y,
        xerr=[
            topic_effects["ame"] - topic_effects["ci_low"],
            topic_effects["ci_high"] - topic_effects["ame"],
        ],
        fmt="none",
        ecolor=PALETTE["ink"],
        elinewidth=1.5,
        capsize=2.5,
        zorder=2,
    )
    ax.scatter(topic_effects["ame"], y, color=colors, s=34, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(topic_effects["topic_label"])
    ax.set_xlabel("Average marginal effect of paper-level sentiment")
    ax.set_title("Topic-cluster heterogeneity in the paper-level model", loc="left", pad=7, fontweight="bold")
    ax.grid(axis="x", alpha=0.35)
    add_panel_label(ax, "A")

    save_figure(fig, "appendix_figure_topic_heterogeneity")


def plot_appendix_legacy_rq1_montage(legacy_assets: dict[str, pd.DataFrame]) -> None:
    descriptive = legacy_assets.get("legacy_descriptive_stats", pd.DataFrame()).copy()
    tests = legacy_assets.get("legacy_univariate_group_tests", pd.DataFrame()).copy()
    multi = legacy_assets.get("legacy_multivariable_logit", pd.DataFrame()).copy()
    if descriptive.empty or tests.empty or multi.empty:
        return

    features = [
        ("mean_sentiment", "Sentiment"),
        ("mean_politeness", "Politeness"),
        ("mean_toxicity", "Toxicity"),
        ("mean_constructiveness", "Constructiveness"),
    ]
    desc_index = descriptive.set_index("decision_label")
    tests = tests.set_index("feature")
    multi = multi.set_index("feature")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.35, 5.95),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1], "height_ratios": [1, 1]},
    )

    x = np.arange(len(features))
    for idx, (feature, label) in enumerate(features):
        reject_mean = float(desc_index.loc["reject", f"{feature}_mean"])
        accept_mean = float(desc_index.loc["accept", f"{feature}_mean"])
        axes[0, 0].plot([idx, idx], [reject_mean, accept_mean], color=PALETTE["muted"], linewidth=1.1, zorder=1)
        axes[0, 0].scatter(idx, reject_mean, color=PALETTE["reject"], s=34, zorder=3)
        axes[0, 0].scatter(idx, accept_mean, color=PALETTE["accept"], s=34, zorder=3)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([label for _, label in features], rotation=18, ha="right")
    axes[0, 0].set_ylabel("Legacy paper-level mean")
    axes[0, 0].set_title("Accepted versus rejected means", loc="left", pad=7, fontweight="bold")
    legacy_means = []
    for feature, _ in features:
        legacy_means.extend(
            [
                float(desc_index.loc["reject", f"{feature}_mean"]),
                float(desc_index.loc["accept", f"{feature}_mean"]),
            ]
        )
    axes[0, 0].set_ylim(min(legacy_means) - 0.03, max(legacy_means) + 0.03)
    axes[0, 0].legend(
        handles=[
            Line2D([0], [0], marker="o", lw=0, color=PALETTE["accept"], label="Accepted"),
            Line2D([0], [0], marker="o", lw=0, color=PALETTE["reject"], label="Rejected"),
        ],
        frameon=False,
        loc="upper right",
    )
    axes[0, 0].grid(axis="y", alpha=0.35)
    add_panel_label(axes[0, 0], "A")

    diff_values = [float(tests.loc[feature, "mean_diff_accept_minus_reject"]) for feature, _ in features]
    bar_colors = [
        PALETTE["accent"] if value >= 0 else PALETTE["reject"]
        for value in diff_values
    ]
    axes[0, 1].axhline(0, color=PALETTE["muted"], linewidth=1.0, linestyle="--")
    axes[0, 1].bar(x, diff_values, color=bar_colors, alpha=0.86, edgecolor="#ffffff", linewidth=0.9)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([label for _, label in features], rotation=18, ha="right")
    axes[0, 1].set_ylabel("Accept - reject mean difference")
    axes[0, 1].set_title("Legacy descriptive gaps", loc="left", pad=7, fontweight="bold")
    axes[0, 1].grid(axis="y", alpha=0.35)
    add_panel_label(axes[0, 1], "B")

    odds = multi.loc[[feature for feature, _ in features]].copy()
    y = np.arange(len(odds))[::-1]
    axes[1, 0].axvline(1.0, color=PALETTE["muted"], linewidth=1.0, linestyle="--")
    axes[1, 0].errorbar(
        odds["odds_ratio"],
        y,
        xerr=[odds["odds_ratio"] - odds["or_ci_low"], odds["or_ci_high"] - odds["odds_ratio"]],
        fmt="o",
        color=PALETTE["accent"],
        ecolor=PALETTE["ink"],
        elinewidth=1.4,
        capsize=2.5,
    )
    axes[1, 0].set_yticks(y)
    axes[1, 0].set_yticklabels([label for _, label in features])
    axes[1, 0].set_xlabel("Odds ratio")
    axes[1, 0].set_title("Legacy logit benchmark", loc="left", pad=7, fontweight="bold")
    axes[1, 0].grid(axis="x", alpha=0.35)
    add_panel_label(axes[1, 0], "C")

    structural = [
        ("num_reviews", "Reviews per paper"),
        ("mean_review_length", "Review length"),
    ]
    x2 = np.arange(len(structural))
    structural_smd = []
    for prefix, _ in structural:
        accept_mean = float(desc_index.loc["accept", f"{prefix}_mean"])
        reject_mean = float(desc_index.loc["reject", f"{prefix}_mean"])
        accept_std = float(desc_index.loc["accept", f"{prefix}_std"])
        reject_std = float(desc_index.loc["reject", f"{prefix}_std"])
        pooled_sd = math.sqrt((accept_std**2 + reject_std**2) / 2.0)
        structural_smd.append((accept_mean - reject_mean) / pooled_sd if pooled_sd else 0.0)
    axes[1, 1].axhline(0, color=PALETTE["muted"], linewidth=1.0, linestyle="--")
    axes[1, 1].bar(
        x2,
        structural_smd,
        color=[PALETTE["accent"] if value >= 0 else PALETTE["reject"] for value in structural_smd],
        alpha=0.86,
        edgecolor="#ffffff",
        linewidth=0.9,
    )
    axes[1, 1].set_xticks(x2)
    axes[1, 1].set_xticklabels([label for _, label in structural], rotation=10, ha="right")
    axes[1, 1].set_ylabel("Standardized mean difference")
    axes[1, 1].set_title("Structural differences", loc="left", pad=7, fontweight="bold")
    axes[1, 1].grid(axis="y", alpha=0.35)
    add_panel_label(axes[1, 1], "D")

    save_figure(fig, "appendix_figure_legacy_rq1")


def plot_appendix_legacy_rq3_montage(legacy_assets: dict[str, pd.DataFrame]) -> None:
    by_year = legacy_assets.get("legacy_descriptive_by_year", pd.DataFrame()).copy()
    by_disagreement = legacy_assets.get("legacy_descriptive_by_disagreement", pd.DataFrame()).copy()
    year_model = legacy_assets.get("legacy_stratified_logit_by_year", pd.DataFrame()).copy()
    if by_year.empty or by_disagreement.empty or year_model.empty:
        return

    sentiment_coef = year_model[year_model["feature"] == "mean_sentiment"].copy().sort_values("year")
    by_year = by_year.sort_values("year")
    by_disagreement["disagreement_level"] = pd.Categorical(
        by_disagreement["disagreement_level"],
        categories=["low", "medium", "high"],
        ordered=True,
    )
    by_disagreement = by_disagreement.sort_values("disagreement_level")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.35, 5.95),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1], "height_ratios": [1, 1]},
    )
    years = by_year["year"].astype(int).tolist()
    x = np.arange(len(years))

    axes[0, 0].plot(x, by_year["acceptance_rate"], color=PALETTE["accent"], marker="o", linewidth=2.0)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(years)
    axes[0, 0].set_ylim(0.25, 0.45)
    axes[0, 0].set_ylabel("Acceptance rate")
    axes[0, 0].set_title("Acceptance rate by year", loc="left", pad=7, fontweight="bold")
    axes[0, 0].grid(axis="y", alpha=0.35)
    add_panel_label(axes[0, 0], "A")

    axes[0, 1].plot(
        x,
        by_year["mean_sentiment_accept_mean"],
        color=PALETTE["accept"],
        marker="o",
        linewidth=2.0,
        label="Accepted",
    )
    axes[0, 1].plot(
        x,
        by_year["mean_sentiment_reject_mean"],
        color=PALETTE["reject"],
        marker="o",
        linewidth=2.0,
        label="Rejected",
    )
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(years)
    axes[0, 1].set_ylabel("Legacy mean sentiment")
    axes[0, 1].set_title("Sentiment by year and decision", loc="left", pad=7, fontweight="bold")
    sentiment_floor = float(
        min(by_year["mean_sentiment_accept_mean"].min(), by_year["mean_sentiment_reject_mean"].min())
    )
    sentiment_ceiling = float(
        max(by_year["mean_sentiment_accept_mean"].max(), by_year["mean_sentiment_reject_mean"].max())
    )
    axes[0, 1].set_ylim(sentiment_floor - 0.012, sentiment_ceiling + 0.03)
    axes[0, 1].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.03, 0.99), borderaxespad=0.0)
    axes[0, 1].grid(axis="y", alpha=0.35)
    add_panel_label(axes[0, 1], "B")

    y = np.arange(len(sentiment_coef))[::-1]
    axes[1, 0].axvline(0, color=PALETTE["muted"], linewidth=1.0, linestyle="--")
    axes[1, 0].errorbar(
        sentiment_coef["coef"],
        y,
        xerr=[sentiment_coef["coef"] - sentiment_coef["ci_low"], sentiment_coef["ci_high"] - sentiment_coef["coef"]],
        fmt="o",
        color=PALETTE["accent"],
        ecolor=PALETTE["ink"],
        elinewidth=1.4,
        capsize=2.5,
    )
    axes[1, 0].set_yticks(y)
    axes[1, 0].set_yticklabels(sentiment_coef["year"].astype(int).astype(str))
    axes[1, 0].set_xlabel("Legacy sentiment coefficient")
    axes[1, 0].set_title("Year-specific sentiment coefficients", loc="left", pad=7, fontweight="bold")
    axes[1, 0].grid(axis="x", alpha=0.35)
    add_panel_label(axes[1, 0], "C")

    x2 = np.arange(len(by_disagreement))
    axes[1, 1].bar(
        x2,
        by_disagreement["mean_sentiment_diff"],
        color=[blend_colors(PALETTE["accent"], "#ffffff", 0.12)] * len(by_disagreement),
        edgecolor="#ffffff",
        linewidth=1.0,
    )
    axes[1, 1].set_xticks(x2)
    axes[1, 1].set_xticklabels([str(value).title() for value in by_disagreement["disagreement_level"]])
    axes[1, 1].set_ylabel("Accept - reject sentiment")
    axes[1, 1].set_title("Sentiment gaps by disagreement", loc="left", pad=7, fontweight="bold")
    axes[1, 1].grid(axis="y", alpha=0.35)
    add_panel_label(axes[1, 1], "D")

    save_figure(fig, "appendix_figure_legacy_rq3")


def format_number(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    return f"{value:.{digits}f}"


def write_numbers_tex(
    paper_df: pd.DataFrame,
    observational: dict[str, pd.DataFrame],
    match_results: dict[str, object],
    prediction_df: pd.DataFrame,
    year_difference_df: pd.DataFrame,
) -> None:
    sentiment_by_accept = paper_df.groupby("accept")["mean_sentiment"].mean()
    politeness_by_accept = paper_df.groupby("accept")["mean_politeness"].mean()
    toxicity_by_accept = paper_df.groupby("accept")["mean_toxicity"].mean()
    constructiveness_by_accept = paper_df.groupby("accept")["mean_constructiveness"].mean()
    score_by_accept = paper_df.groupby("accept")["mean_score"].mean()
    review_length_by_accept = paper_df.groupby("accept")["review_length_mean"].mean()
    paper_ame = observational["paper_ame"].set_index("term").loc["mean_sentiment"]
    review_ame = observational["review_ame"].set_index("term").loc["sentiment"]
    review_politeness_ame = observational["review_ame"].set_index("term").loc["politeness"]
    review_constructiveness_ame = observational["review_ame"].set_index("term").loc["constructiveness"]
    paper_margins = observational["paper_margins"]
    p10_value = float(paper_df["mean_sentiment"].quantile(0.10))
    p90_value = float(paper_df["mean_sentiment"].quantile(0.90))
    p10_pred = float(paper_margins.iloc[(paper_margins["mean_sentiment"] - p10_value).abs().argmin()]["predicted_probability"])
    p90_pred = float(paper_margins.iloc[(paper_margins["mean_sentiment"] - p90_value).abs().argmin()]["predicted_probability"])
    primary_diag = match_results["primary_diag"]
    narrow_diag = match_results["narrow_diag"]
    wide_diag = match_results["wide_diag"]
    robust_diag = match_results["robust_diag"]
    hetero_year = observational["heterogeneity_year"]
    hetero_disagreement = observational["heterogeneity_disagreement"]
    yearly_sentiment = year_difference_df[year_difference_df["feature"] == "mean_sentiment"]
    yearly_politeness = year_difference_df[year_difference_df["feature"] == "mean_politeness"]
    commands = {
        "TotalPapers": f"{len(paper_df):,}",
        "TotalReviews": f"{int(paper_df['num_reviews'].sum()):,}",
        "AcceptedPapers": f"{int(paper_df['accept'].sum()):,}",
        "RejectedPapers": f"{int((1 - paper_df['accept']).sum()):,}",
        "OverallAcceptancePct": format_number(100 * paper_df["accept"].mean(), 1),
        "SentimentAcceptMean": format_number(float(sentiment_by_accept.loc[1]), 3),
        "SentimentRejectMean": format_number(float(sentiment_by_accept.loc[0]), 3),
        "PolitenessAcceptMean": format_number(float(politeness_by_accept.loc[1]), 3),
        "PolitenessRejectMean": format_number(float(politeness_by_accept.loc[0]), 3),
        "ToxicityAcceptMean": format_number(float(toxicity_by_accept.loc[1]), 3),
        "ToxicityRejectMean": format_number(float(toxicity_by_accept.loc[0]), 3),
        "ConstructivenessAcceptMean": format_number(float(constructiveness_by_accept.loc[1]), 3),
        "ConstructivenessRejectMean": format_number(float(constructiveness_by_accept.loc[0]), 3),
        "MeanScoreAccept": format_number(float(score_by_accept.loc[1]), 2),
        "MeanScoreReject": format_number(float(score_by_accept.loc[0]), 2),
        "ReviewLengthAccept": format_number(float(review_length_by_accept.loc[1]), 1),
        "ReviewLengthReject": format_number(float(review_length_by_accept.loc[0]), 1),
        "PaperSentimentAME": format_number(float(paper_ame["ame"]), 3),
        "PaperSentimentAMECI": f"[{format_number(float(paper_ame['ci_low']), 3)}, {format_number(float(paper_ame['ci_high']), 3)}]",
        "ReviewSentimentAME": format_number(float(review_ame["ame"]), 3),
        "ReviewSentimentAMECI": f"[{format_number(float(review_ame['ci_low']), 3)}, {format_number(float(review_ame['ci_high']), 3)}]",
        "ReviewPolitenessAME": format_number(float(review_politeness_ame["ame"]), 3),
        "ReviewPolitenessAMECI": f"[{format_number(float(review_politeness_ame['ci_low']), 3)}, {format_number(float(review_politeness_ame['ci_high']), 3)}]",
        "ReviewConstructivenessAME": format_number(float(review_constructiveness_ame["ame"]), 3),
        "ReviewConstructivenessAMECI": f"[{format_number(float(review_constructiveness_ame['ci_low']), 3)}, {format_number(float(review_constructiveness_ame['ci_high']), 3)}]",
        "PredProbLowSent": format_number(p10_pred, 3),
        "PredProbHighSent": format_number(p90_pred, 3),
        "BaselineAUCMin": format_number(float(prediction_df["auc_baseline"].min()), 3),
        "BaselineAUCMax": format_number(float(prediction_df["auc_baseline"].max()), 3),
        "MaxAbsDeltaAUC": format_number(float(prediction_df["delta_auc"].abs().max()), 4),
        "YearSentimentDiffMin": format_number(float(yearly_sentiment["effect"].min()), 3),
        "YearSentimentDiffMax": format_number(float(yearly_sentiment["effect"].max()), 3),
        "YearPolitenessDiffMin": format_number(float(yearly_politeness["effect"].min()), 3),
        "YearPolitenessDiffMax": format_number(float(yearly_politeness["effect"].max()), 3),
        "BorderlinePapers": f"{len(match_results['borderline_sample']):,}",
        "MatchedPairs": f"{int(primary_diag['n_matched_pairs']):,}",
        "PSMATT": format_number(float(primary_diag["att"]), 3),
        "PSMATTCI": f"[{format_number(float(primary_diag['ci_low']), 3)}, {format_number(float(primary_diag['ci_high']), 3)}]",
        "PSMNarrowATT": format_number(float(narrow_diag["att"]), 3),
        "PSMNarrowATTCI": f"[{format_number(float(narrow_diag['ci_low']), 3)}, {format_number(float(narrow_diag['ci_high']), 3)}]",
        "PSMWideATT": format_number(float(wide_diag["att"]), 3),
        "PSMWideATTCI": f"[{format_number(float(wide_diag['ci_low']), 3)}, {format_number(float(wide_diag['ci_high']), 3)}]",
        "PSMRobustATT": format_number(float(robust_diag["att"]), 3),
        "PSMRobustATTCI": f"[{format_number(float(robust_diag['ci_low']), 3)}, {format_number(float(robust_diag['ci_high']), 3)}]",
        "BalanceStatus": "passed" if primary_diag["balance_pass"] else "flagged",
        "ClusterCount": f"{paper_df['topic_cluster'].nunique()}",
        "HeteroYearMin": format_number(float(hetero_year["ame"].min()), 3),
        "HeteroYearMax": format_number(float(hetero_year["ame"].max()), 3),
        "HeteroDisagreementMin": format_number(float(hetero_disagreement["ame"].min()), 3),
        "HeteroDisagreementMax": format_number(float(hetero_disagreement["ame"].max()), 3),
    }
    lines = ["% Auto-generated by scripts/build_causal_package.py"]
    for key, value in commands.items():
        lines.append(f"\\newcommand{{\\{key}}}{{{value}}}")
    NUMBERS_PATH.write_text("\n".join(lines) + "\n")


def latex_escape(text: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def latex_figure_from_file(filename: str, caption: str, label: str, width: str = "0.98\\linewidth") -> str:
    lines = [
        "\\begin{figure}[!htbp]",
        "\\centering",
        f"\\includegraphics[width={width}]{{{filename}}}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{figure}",
        "",
    ]
    return "\n".join(lines)


def latex_table_from_df(
    df: pd.DataFrame,
    columns: list[str],
    caption: str,
    label: str,
    align: str | None = None,
    float_fmt: dict[str, int] | None = None,
    size: str | None = None,
    resize_to_width: bool = False,
) -> str:
    align = align or ("l" + "r" * (len(columns) - 1))
    float_fmt = float_fmt or {}
    size = size or "normalsize"
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\begingroup",
        f"\\{size}",
        "\\renewcommand{\\arraystretch}{1.20}",
        "\\setlength{\\tabcolsep}{6pt}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
    ]
    if resize_to_width:
        lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.extend([
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(latex_escape(column) for column in columns) + " \\\\",
        "\\midrule",
    ])
    for row in df[columns].to_dict(orient="records"):
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                digits = float_fmt.get(column, 3)
                values.append(format_number(float(value), digits))
            elif isinstance(value, (int, np.integer)):
                values.append(str(int(value)))
            else:
                values.append(latex_escape(value))
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    if resize_to_width:
        lines.append("}")
    lines.extend(["\\endgroup", "\\end{table}", ""])
    return "\n".join(lines)


def cast_display_ints(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="raise").round().astype(int)
    return out


def write_appendix_tables(
    paper_df: pd.DataFrame,
    review_df: pd.DataFrame,
    topic_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    observational: dict[str, pd.DataFrame],
    match_results: dict[str, object],
    prediction_df: pd.DataFrame,
    year_difference_df: pd.DataFrame,
    measurement_year_df: pd.DataFrame,
    bridge_df: pd.DataFrame,
    legacy_assets: dict[str, pd.DataFrame],
) -> None:
    year_table = audit_df.groupby("year", as_index=False).agg(
        papers=("forum", "nunique"),
        parsed_reviews=("parsed_reviews", "sum"),
        decision_coverage=("has_decision", "sum"),
    )
    year_table["accept_rate"] = paper_df.groupby("year")["accept"].mean().values
    year_table["rating_parse_rate"] = review_df.groupby("year")["rating_parse_success"].mean().values
    year_table["confidence_parse_rate"] = review_df.groupby("year")["confidence_parse_success"].mean().values
    year_table["decision_parse_rate"] = paper_df.groupby("year")["decision_parse_success"].mean().values
    year_table["score_source"] = ["rating", "rating", "rating", "rating", "recommendation", "recommendation"]
    year_table["text_source"] = [
        "review",
        "review",
        "review",
        "review",
        "summary_of_the_paper + summary_of_the_review",
        "summary_of_the_paper + strength_and_weaknesses + summary_of_the_review",
    ]

    paper_ame = observational["paper_ame"].copy()
    paper_ame = paper_ame[paper_ame["term"] != "confidence_missing"].copy()
    review_ame = observational["review_ame"].copy()
    heterogeneity_year = observational["heterogeneity_year"].copy()
    heterogeneity_disagreement = observational["heterogeneity_disagreement"].copy()
    heterogeneity_topic = observational["heterogeneity_topic"].copy()
    primary_balance = match_results["primary_balance"].copy()
    topic_short = topic_df.copy()
    topic_short["top_terms"] = topic_short["top_terms"].map(lambda s: textwrap.shorten(s, width=38, placeholder="..."))
    primary_counts = match_results["primary_counts"].copy()
    effects = match_results["effects"].copy()
    effects["balance_pass"] = effects["balance_pass"].map(lambda value: "Pass" if bool(value) else "Flag")
    overlap = match_results["overlap"].copy()
    overlap["balance_pass"] = overlap["balance_pass"].map(lambda value: "Pass" if bool(value) else "Flag")
    spec_label_map = {
        "Primary window 35th-65th percentile": "Primary 35-65 window",
        "Window sensitivity 40th-60th percentile": "Window 40-60",
        "Window sensitivity 30th-70th percentile": "Window 30-70",
        "Treatment sensitivity: top vs bottom sentiment tercile": "Top vs bottom terciles",
    }
    effects["specification"] = effects["specification"].map(lambda value: spec_label_map.get(value, value))
    overlap["specification"] = overlap["specification"].map(lambda value: spec_label_map.get(value, value))
    fit_note_map = {
        "requested_spec": "Requested specification",
        "fallback_hc1_cov": "HC1 covariance fallback",
        "fallback_without_fixed_effects": "Fixed effects relaxed",
    }
    heterogeneity_year["fit_note"] = heterogeneity_year["fit_note"].map(lambda value: fit_note_map.get(value, value))
    heterogeneity_disagreement["fit_note"] = heterogeneity_disagreement["fit_note"].map(
        lambda value: fit_note_map.get(value, value)
    )
    heterogeneity_topic["fit_note"] = heterogeneity_topic["fit_note"].map(lambda value: fit_note_map.get(value, value))
    paper_label_map = {
        "Paper-level mean review sentiment": "Paper-level mean review sentiment",
        "Paper-level mean review politeness": "Paper-level mean review politeness",
        "Paper-level mean review toxicity": "Paper-level mean review toxicity",
        "Paper-level mean review constructiveness": "Paper-level mean review constructiveness",
        "Mean reviewer score": "Mean reviewer score",
        "Reviewer score disagreement": "Reviewer score disagreement",
        "Reviewer confidence": "Mean reviewer confidence",
        "Number of reviews": "Number of reviews",
        "Mean review length (tokens)": "Mean review length (tokens)",
        "Title length (tokens)": "Title length (tokens)",
        "Abstract length (tokens)": "Abstract length (tokens)",
        "Number of keywords": "Number of keywords",
        "Confidence missingness indicator": "Confidence missingness indicator",
    }
    review_label_map = {
        "Review-level sentiment": "Review sentiment",
        "Review-level politeness": "Review politeness",
        "Review-level toxicity": "Review toxicity",
        "Review-level constructiveness": "Review constructiveness",
        "Reviewer confidence": "Reviewer confidence",
        "Review length (tokens)": "Review length (tokens)",
        "Title length (tokens)": "Title length (tokens)",
        "Abstract length (tokens)": "Abstract length (tokens)",
        "Number of keywords": "Number of keywords",
        "Confidence missingness indicator": "Confidence missingness indicator",
    }
    lexicon_label_map = {
        "sentiment_pos_words": "Positive sentiment markers",
        "sentiment_neg_words": "Negative sentiment markers",
        "afinn_scores": "AFINN sentiment scores",
        "politeness_pos_words": "Positive politeness markers",
        "politeness_neg_words": "Negative politeness markers",
        "politeness_pos_phrases": "Positive politeness phrases",
        "politeness_neg_phrases": "Negative politeness phrases",
        "toxicity_words": "Toxicity markers",
        "toxic_words": "Toxicity words",
        "toxic_phrases": "Toxicity phrases",
        "constructive_words": "Constructive feedback markers",
        "constructive_phrases": "Constructive feedback phrases",
        "hedge_words": "Hedging words",
        "hedge_phrases": "Hedging phrases",
    }
    score_source_map = {"rating": "Rating", "recommendation": "Recommendation"}
    text_source_map = {
        "review": "Review text",
        "summary_of_the_paper + summary_of_the_review": "Paper summary + review summary",
        "summary_of_the_paper + strength_and_weaknesses + summary_of_the_review": (
            "Paper summary + strengths and weaknesses + review summary"
        ),
    }

    measurement_display = measurement_year_df.rename(
        columns={
            "year": "Year",
            "papers": "Submissions",
            "reviews": "Reviews",
            "acceptance_rate": "Acceptance rate",
            "positive_recommend_share": "Positive recommendation share",
            "confidence_parse_rate": "Confidence parse rate",
            "keyword_missing_share": "Keyword missingness",
            "mean_review_length": "Mean review length (tokens)",
        }
    ).copy()
    measurement_display = cast_display_ints(measurement_display, ["Year", "Submissions", "Reviews"])

    year_table_display = year_table.rename(
        columns={
            "year": "Year",
            "papers": "Submissions",
            "parsed_reviews": "Reviews",
            "decision_coverage": "Final decisions",
            "accept_rate": "Acceptance rate",
            "rating_parse_rate": "Score parse rate",
            "confidence_parse_rate": "Confidence parse rate",
            "decision_parse_rate": "Decision parse rate",
            "score_source": "Score field",
            "text_source": "Review-text field",
        }
    ).copy()
    year_table_display["Score field"] = year_table_display["Score field"].map(
        lambda value: score_source_map.get(value, value)
    )
    year_table_display["Review-text field"] = year_table_display["Review-text field"].map(
        lambda value: text_source_map.get(value, value)
    )
    year_table_display = cast_display_ints(
        year_table_display,
        ["Year", "Submissions", "Reviews", "Final decisions"],
    )

    paper_ame_display = paper_ame.rename(
        columns={
            "label": "Predictor",
            "ame": "Average marginal effect",
            "ci_low": "95% CI lower",
            "ci_high": "95% CI upper",
        }
    ).copy()
    paper_ame_display["Predictor"] = paper_ame_display["Predictor"].map(
        lambda value: paper_label_map.get(value, value)
    )
    review_ame_display = review_ame.rename(
        columns={
            "label": "Predictor",
            "ame": "Average marginal effect",
            "ci_low": "95% CI lower",
            "ci_high": "95% CI upper",
        }
    ).copy()
    review_ame_display["Predictor"] = review_ame_display["Predictor"].map(
        lambda value: review_label_map.get(value, value)
    )
    yearly_gaps = year_difference_df[
        year_difference_df["feature"].isin(["mean_sentiment", "mean_politeness"])
    ].copy()
    yearly_gaps["feature"] = yearly_gaps["feature"].map(
        {
            "mean_sentiment": "Sentiment gap",
            "mean_politeness": "Politeness gap",
        }
    )
    yearly_gaps_display = yearly_gaps.rename(
        columns={
            "feature": "Language dimension",
            "year": "Year",
            "effect": "Mean difference",
            "ci_low": "95% CI lower",
            "ci_high": "95% CI upper",
        }
    ).copy()
    yearly_gaps_display = cast_display_ints(yearly_gaps_display, ["Year"])
    heterogeneity_topic = heterogeneity_topic.merge(
        topic_short[["topic_cluster", "top_terms"]],
        on="topic_cluster",
        how="left",
    )
    heterogeneity_topic["topic_cluster"] = heterogeneity_topic["topic_cluster"].astype(int)
    heterogeneity_year_display = heterogeneity_year.rename(
        columns={
            "year": "Year",
            "ame": "Average marginal effect",
            "ci_low": "95% CI lower",
            "ci_high": "95% CI upper",
            "fit_note": "Specification note",
        }
    ).copy()
    heterogeneity_year_display = cast_display_ints(heterogeneity_year_display, ["Year"])
    heterogeneity_disagreement_display = heterogeneity_disagreement.rename(
        columns={
            "disagreement_group": "Reviewer disagreement tier",
            "ame": "Average marginal effect",
            "ci_low": "95% CI lower",
            "ci_high": "95% CI upper",
            "fit_note": "Specification note",
        }
    ).copy()
    heterogeneity_topic_display = heterogeneity_topic.rename(
        columns={
            "topic_cluster": "Topic cluster",
            "n_papers": "Submissions",
            "ame": "Average marginal effect",
            "ci_low": "95% CI lower",
            "ci_high": "95% CI upper",
            "top_terms": "Representative keywords",
        }
    ).copy()
    heterogeneity_topic_display = cast_display_ints(heterogeneity_topic_display, ["Topic cluster", "Submissions"])

    bridge_table = bridge_df.copy()
    bridge_table = bridge_table.rename(
        columns={
            "score_bin": "Score decile",
            "n_papers": "Submissions",
            "n_accept": "Accepted submissions",
            "n_reject": "Rejected submissions",
            "acceptance_rate": "Acceptance rate",
            "sentiment_gap": "Sentiment difference",
            "ci_low": "95% CI lower",
            "ci_high": "95% CI upper",
        }
    ).copy()
    bridge_table = cast_display_ints(bridge_table, ["Submissions", "Accepted submissions", "Rejected submissions"])

    lexicon_summary = legacy_assets.get("lexicon_summary", pd.DataFrame()).copy()
    lexicon_display = lexicon_summary.rename(columns={"lexicon": "Lexicon", "size": "Entries"}).copy()
    if not lexicon_display.empty:
        lexicon_display["Lexicon"] = lexicon_display["Lexicon"].map(lambda value: lexicon_label_map.get(value, value))
        lexicon_display = cast_display_ints(lexicon_display, ["Entries"])
    legacy_multivariable = legacy_assets.get("legacy_multivariable_logit", pd.DataFrame()).copy()
    if not legacy_multivariable.empty:
        legacy_multivariable = legacy_multivariable[
            legacy_multivariable["feature"].isin(
                [
                    "mean_sentiment",
                    "mean_politeness",
                    "mean_toxicity",
                    "mean_constructiveness",
                    "num_reviews",
                    "mean_review_length",
                ]
            )
        ].copy()
        legacy_multivariable["feature"] = legacy_multivariable["feature"].map(
            {
                "mean_sentiment": "Legacy sentiment",
                "mean_politeness": "Legacy politeness",
                "mean_toxicity": "Legacy toxicity",
                "mean_constructiveness": "Legacy constructiveness",
                "num_reviews": "Legacy reviews per paper",
                "mean_review_length": "Legacy review length",
            }
        )
    legacy_multivariable_display = legacy_multivariable.rename(
        columns={
            "feature": "Predictor",
            "odds_ratio": "Odds ratio",
            "or_ci_low": "95% CI lower",
            "or_ci_high": "95% CI upper",
            "pvalue": "P-value",
        }
    ).copy()
    legacy_meta = legacy_assets.get("legacy_meta_analysis", pd.DataFrame()).copy()
    if not legacy_meta.empty:
        legacy_meta = legacy_meta[
            ["feature", "pooled_effect_fe", "I2_percent", "stability_interpretation"]
        ].copy()
        legacy_meta["feature"] = legacy_meta["feature"].map(
            {
                "mean_sentiment": "Legacy sentiment",
                "mean_politeness": "Legacy politeness",
                "mean_toxicity": "Legacy toxicity",
                "mean_constructiveness": "Legacy constructiveness",
            }
        )
    legacy_meta_display = legacy_meta.rename(
        columns={
            "feature": "Measure",
            "pooled_effect_fe": "Pooled effect",
            "I2_percent": "I-squared (%)",
            "stability_interpretation": "Stability pattern",
        }
    ).copy()
    effects_display = effects.rename(
        columns={
            "specification": "Matching specification",
            "att": "Matched contrast (ATT)",
            "ci_low": "95% CI lower",
            "ci_high": "95% CI upper",
            "matched_pairs": "Matched pairs",
            "balance_pass": "Balance target met",
        }
    ).copy()
    effects_display = cast_display_ints(effects_display, ["Matched pairs"])
    if "Balance target met" in effects_display.columns:
        effects_display["Balance target met"] = effects_display["Balance target met"].map(
            lambda value: "Yes" if bool(value) else "No"
        )
    primary_balance_display = primary_balance.rename(
        columns={
            "label": "Covariate",
            "smd_before": "Absolute SMD before matching",
            "smd_after": "Absolute SMD after matching",
        }
    ).copy()
    overlap_display = overlap.rename(
        columns={
            "specification": "Matching specification",
            "n_analytic": "Analytic sample size",
            "n_treated": "Higher-sentiment papers",
            "n_control": "Lower-sentiment papers",
            "n_matched_pairs": "Matched pairs",
            "overlap_lower": "Common-support lower bound",
            "overlap_upper": "Common-support upper bound",
            "caliper": "Caliper",
            "balance_pass": "Balance target met",
        }
    ).copy()
    overlap_display = cast_display_ints(
        overlap_display,
        ["Analytic sample size", "Higher-sentiment papers", "Lower-sentiment papers", "Matched pairs"],
    )
    if "Balance target met" in overlap_display.columns:
        overlap_display["Balance target met"] = overlap_display["Balance target met"].map(
            lambda value: "Yes" if bool(value) else "No"
        )
    primary_counts_display = primary_counts.rename(columns={"year": "Year", "matched_pairs": "Matched pairs"}).copy()
    primary_counts_display = cast_display_ints(primary_counts_display, ["Year", "Matched pairs"])
    topic_short_display = topic_short.rename(
        columns={"topic_cluster": "Topic cluster", "n_papers": "Submissions", "top_terms": "Representative keywords"}
    ).copy()
    topic_short_display = cast_display_ints(topic_short_display, ["Topic cluster", "Submissions"])
    prediction_display = prediction_df.rename(
        columns={
            "heldout_year": "Held-out year",
            "auc_baseline": "Baseline AUC",
            "auc_with_language": "AUC with language",
            "delta_auc": "AUC change",
            "delta_log_loss": "Log-loss change",
            "delta_brier": "Brier-score change",
        }
    ).copy()
    prediction_display = cast_display_ints(prediction_display, ["Held-out year"])

    sections = [
        "\\FloatBarrier",
        "\\onecolumn",
        "\\appendix",
        "\\renewcommand{\\thetable}{A\\arabic{table}}",
        "\\setcounter{table}{0}",
        "\\renewcommand{\\thefigure}{A\\arabic{figure}}",
        "\\setcounter{figure}{0}",
        "\\section*{Appendix: Archive Reconstruction and Measurement Checks}",
        (
            "This appendix documents the reconstruction of the raw ICLR archive and the "
            "measurement checks that bound interpretation of the main-text results. It "
            "reports archive coverage, parser completeness, field harmonization, and "
            "measurement infrastructure before turning to observational, matched-sample, "
            "and legacy supplementary materials."
        ),
        "",
        latex_figure_from_file(
            "appendix_figure_measurement_summary.pdf",
            (
                "Archive coverage and measurement checks for the reconstructed ICLR review "
                "corpus. \\textbf{A}, yearly paper and review counts in the harmonized "
                "archive. \\textbf{B}, confidence parsing, keyword coverage, and positive-"
                "review prevalence by year; the most visible discontinuity is the loss of "
                "a directly comparable confidence field in 2020. \\textbf{C}, the "
                "distribution of reviews per paper, showing that most submissions receive "
                "three to four reviews. \\textbf{D}, review-length distributions by final "
                "decision. Together, the panels establish the scale of the archive and the "
                "practical measurement boundaries within which the main-text models should "
                "be interpreted."
            ),
            "fig:app:measurement",
        ),
        latex_figure_from_file(
            "appendix_figure_feature_correlations.pdf",
            (
                "Paper-level correlation structure linking language measures to the broader "
                "evaluation profile. The matrix shows that sentiment co-moves strongly with "
                "reviewer score and more modestly with review length, whereas toxicity is "
                "only weakly related to most other observables. The inferential point is "
                "that review language does not float outside the rest of the archive: it is "
                "embedded in the same profile that also contains scores, confidence, and "
                "review structure."
            ),
            "fig:app:correlations",
        ),
        latex_table_from_df(
            year_table_display,
            [
                "Year",
                "Submissions",
                "Reviews",
                "Final decisions",
                "Acceptance rate",
                "Score parse rate",
                "Confidence parse rate",
                "Decision parse rate",
            ],
            (
                "Raw-archive parsing diagnostics by year. The builder retains raw strings, "
                "parsed numeric fields, and parse-success flags for ratings, confidence, and "
                "final decisions. Parsing succeeds nearly completely for scores and decisions "
                "across all years, whereas confidence is the clearest year-specific break "
                "because the 2020 export does not preserve a directly comparable field."
            ),
            "tab:app:harmonization",
            align="rrrrrrrr",
            float_fmt={
                "Acceptance rate": 3,
                "Score parse rate": 3,
                "Confidence parse rate": 3,
                "Decision parse rate": 3,
            },
            size="scriptsize",
        ),
        latex_table_from_df(
            year_table_display,
            ["Year", "Score field", "Review-text field"],
            (
                "Year-specific score and review-text sources used in the harmonized raw-"
                "archive parser. The table records the field substitutions required to make "
                "the archive comparable across years, especially the shift from "
                "\\texttt{rating} to \\texttt{recommendation} and the richer multi-field "
                "review text available in 2022--2023."
            ),
            "tab:app:fieldmap",
            align="rll",
        ),
        latex_table_from_df(
            measurement_display,
            [
                "Year",
                "Submissions",
                "Reviews",
                "Acceptance rate",
                "Positive recommendation share",
                "Confidence parse rate",
                "Keyword missingness",
                "Mean review length (tokens)",
            ],
            (
                "Measurement summary by year. The table reports archive scale, positive "
                "recommendation prevalence, parser completeness, manuscript-metadata "
                "missingness, and average review length. It shows that the reconstructed "
                "archive is large and broadly comparable across years, while also clarifying "
                "that 2022 uses shorter review-text fields on average and that keyword "
                "coverage, though imperfect, remains usable for topic-proxy construction."
            ),
            "tab:app:measurementyear",
            align="rrrrrrrr",
            float_fmt={
                "Acceptance rate": 3,
                "Positive recommendation share": 3,
                "Confidence parse rate": 3,
                "Keyword missingness": 3,
                "Mean review length (tokens)": 3,
            },
            size="scriptsize",
            resize_to_width=True,
        ),
        latex_table_from_df(
            lexicon_display if not lexicon_display.empty else pd.DataFrame({"Lexicon": [], "Entries": []}),
            ["Lexicon", "Entries"],
            (
                "Legacy lexicon inventory carried forward from the earlier exploratory "
                "workflow. These resources are retained as measurement background rather "
                "than as the manuscript's source of truth. Including them here clarifies "
                "what the archive-first rebuild inherited, audited, and simplified."
            ),
            "tab:app:lexicon",
            align="lr",
        ),
        "\\FloatBarrier",
        "\\vspace{0.6em}",
        "\\section*{Appendix: Descriptive Stability and Conditional Diagnostics}",
        (
            "The next set of appendix materials expands the main observational story. It "
            "shows how accepted-minus-rejected language gaps behave across years and score "
            "regions, and how the conditional paper-level sentiment association varies "
            "across disagreement groups and topic-cluster proxies."
        ),
        "",
        latex_figure_from_file(
            "figure3_temporal_stability.pdf",
            (
                "Temporal stability of the descriptive language gap. Accepted-minus-rejected "
                "differences are shown separately for sentiment and politeness in each "
                "conference year, with pooled summaries below the year-specific estimates. "
                "The sign of the gap does not reverse in any year for either measure, which "
                "is why the main text treats descriptive sentiment and politeness alignment "
                "with acceptance as the most stable empirical regularity in the study."
            ),
            "fig:app:temporal",
        ),
        latex_figure_from_file(
            "appendix_figure_score_bridge.pdf",
            (
                "Bridge analysis linking the full-sample descriptive layer to the matched "
                "borderline layer. \\textbf{A}, acceptance rates across within-year "
                "mean-score deciles, with bar color tracking the decision mix of each bin. "
                "\\textbf{B}, accepted-minus-rejected sentiment gaps within the same score "
                "bins. The central pattern is attenuation in the middle of the score "
                "distribution, where accepted and rejected papers overlap most plausibly; "
                "the largest raw differences occur instead in regions where decisions are "
                "already close to deterministic."
            ),
            "fig:app:bridge",
        ),
        latex_figure_from_file(
            "appendix_figure_topic_heterogeneity.pdf",
            (
                "Topic-cluster heterogeneity in the paper-level sentiment association. Topic "
                "controls are keyword-based clusters and should be interpreted as noisy "
                "topical proxies rather than official research areas. Most topic-specific "
                "estimates are imprecise, and none reveals an obvious subfield in which the "
                "conditional paper-level sentiment association becomes consistently large "
                "and positive."
            ),
            "fig:app:topichetero",
        ),
        latex_table_from_df(
            paper_ame_display,
            ["Predictor", "Average marginal effect", "95% CI lower", "95% CI upper"],
            (
                "Average marginal effects from the paper-level fixed-effects model. The "
                "outcome is final acceptance. Full models include the score, disagreement, "
                "confidence, review-structure, manuscript-side, year, and topic controls "
                "described in the main text; this table highlights the language dimensions "
                "because they are central to the paper's substantive argument."
            ),
            "tab:app:paperame",
            align="lrrr",
            float_fmt={"Average marginal effect": 3, "95% CI lower": 3, "95% CI upper": 3},
        ),
        latex_table_from_df(
            review_ame_display,
            ["Predictor", "Average marginal effect", "95% CI lower", "95% CI upper"],
            (
                "Average marginal effects from the review-level fixed-effects model. The "
                "outcome is a positive reviewer recommendation (rating or recommendation "
                "score $\\ge 6$). As in the main text, the table shows that review-level "
                "sentiment has a much larger association with recommendation than any other "
                "tone dimension, which is why the review-level model is treated as the "
                "clearest evidence that language tracks reviewer stance."
            ),
            "tab:app:reviewame",
            "lrrr",
            float_fmt={"Average marginal effect": 3, "95% CI lower": 3, "95% CI upper": 3},
        ),
        latex_table_from_df(
            yearly_gaps_display,
            ["Language dimension", "Year", "Mean difference", "95% CI lower", "95% CI upper"],
            (
                "Accepted-minus-rejected yearly descriptive gaps for the two main language "
                "dimensions discussed in the paper. Sentiment differences are positive in "
                "every year and larger than politeness differences throughout, reinforcing "
                "the interpretation that evaluative valence rather than generic civility is "
                "doing most of the descriptive work."
            ),
            "tab:app:yearlygaps",
            "lrrrr",
            float_fmt={"Mean difference": 3, "95% CI lower": 3, "95% CI upper": 3},
        ),
        latex_table_from_df(
            bridge_table,
            [
                "Score decile",
                "Submissions",
                "Accepted submissions",
                "Rejected submissions",
                "Acceptance rate",
                "Sentiment difference",
                "95% CI lower",
                "95% CI upper",
            ],
            (
                "Within-score-bin bridge diagnostics. Bins are based on the within-year "
                "percentile of paper-level mean score. The table underlies the bridge "
                "figure and shows numerically that the accepted-minus-rejected sentiment "
                "gap is much smaller in the central score bins than in the extreme bins "
                "where acceptance is nearly certain or nearly impossible."
            ),
            "tab:app:bridge",
            "lrrrrrrr",
            float_fmt={
                "Acceptance rate": 3,
                "Sentiment difference": 3,
                "95% CI lower": 3,
                "95% CI upper": 3,
            },
            size="scriptsize",
            resize_to_width=True,
        ),
        latex_table_from_df(
            heterogeneity_year_display,
            ["Year", "Average marginal effect", "95% CI lower", "95% CI upper", "Specification note"],
            (
                "Year-specific average marginal effects of paper-level sentiment in the "
                "paper-level model. The estimates move around zero but do not reveal a year "
                "in which the conditional paper-level sentiment association becomes both "
                "large and precisely positive."
            ),
            "tab:app:heteroyear",
            "rrrrl",
            float_fmt={"Average marginal effect": 3, "95% CI lower": 3, "95% CI upper": 3},
            size="small",
        ),
        latex_table_from_df(
            heterogeneity_disagreement_display,
            ["Reviewer disagreement tier", "Average marginal effect", "95% CI lower", "95% CI upper", "Specification note"],
            (
                "Average marginal effects of paper-level sentiment by disagreement tercile. "
                "This table complements the main heterogeneity figure by showing "
                "numerically that high reviewer disagreement does not produce a markedly "
                "larger positive paper-level sentiment effect."
            ),
            "tab:app:heterodisagreement",
            "lrrrl",
            float_fmt={"Average marginal effect": 3, "95% CI lower": 3, "95% CI upper": 3},
        ),
        latex_table_from_df(
            heterogeneity_topic_display,
            ["Topic cluster", "Submissions", "Average marginal effect", "95% CI lower", "95% CI upper", "Representative keywords"],
            (
                "Average marginal effects of paper-level sentiment by keyword-based topic "
                "cluster. Because the topic clusters are deterministic keyword proxies "
                "rather than official tracks, the table is best read as a boundary check: "
                "it helps rule out the concern that the pooled near-null paper-level "
                "estimate is hiding one dominant topic-specific positive effect."
            ),
            "tab:app:heterotopic",
            "rrr rrl".replace(" ", ""),
            float_fmt={"Average marginal effect": 3, "95% CI lower": 3, "95% CI upper": 3},
        ),
        latex_figure_from_file(
            "appendix_figure_prediction_diagnostics.pdf",
            (
                "Held-out predictive boundary checks. \\textbf{A}, held-out AUC for the "
                "baseline model and the model augmented with paper-level language "
                "features. \\textbf{B}, incremental gains from adding language on held-out "
                "AUC and Brier score by year. Discrimination is already high in the "
                "score-based baseline models, and the incremental changes from adding "
                "language are tiny in absolute magnitude, which is why prediction is "
                "treated as a boundary condition rather than as a headline contribution."
            ),
            "fig:app:prediction",
        ),
        latex_table_from_df(
            prediction_display,
            [
                "Held-out year",
                "Baseline AUC",
                "AUC with language",
                "AUC change",
                "Log-loss change",
                "Brier-score change",
            ],
            (
                "Leave-one-year-out predictive boundary checks. The baseline uses score and "
                "manuscript controls; the extended model adds paper-level language "
                "features. Held-out AUC is already high in the baseline models, and the "
                "incremental changes from adding language are extremely small in absolute "
                "magnitude, which is why prediction is treated as a boundary condition "
                "rather than as a headline contribution."
            ),
            "tab:app:prediction",
            "rrrrrr",
            float_fmt={
                "Baseline AUC": 3,
                "AUC with language": 3,
                "AUC change": 3,
                "Log-loss change": 3,
                "Brier-score change": 3,
            },
        ),
        "\\FloatBarrier",
        "\\vspace{0.6em}",
        "\\section*{Appendix: Matched Borderline Diagnostics}",
        (
            "These appendix materials document common support, balance, matched-sample "
            "composition, and sensitivity checks for the borderline design. Their role is "
            "to show that the near-null matched estimates arise in a sample with improved "
            "comparability rather than from a visibly broken match."
        ),
        "",
        latex_figure_from_file(
            "appendix_figure_psm_overlap.pdf",
            (
                "Propensity-score overlap for the primary borderline specification before "
                "and after matching. The matched sample retains substantial overlap while "
                "sharpening comparability between higher-sentiment and lower-sentiment "
                "papers. This diagnostic matters because it shows that the near-zero "
                "matched estimate is not being produced by an obviously broken common-"
                "support region."
            ),
            "fig:app:overlap",
        ),
        latex_table_from_df(
            effects_display,
            [
                "Matching specification",
                "Matched contrast (ATT)",
                "95% CI lower",
                "95% CI upper",
                "Matched pairs",
                "Balance target met",
            ],
            (
                "Matched observational evidence from the borderline layer. The table "
                "reports the primary 35th-65th percentile window, two prespecified window "
                "sensitivities, and the top-versus-bottom sentiment-tercile sensitivity "
                "check. Across all four designs, estimated sentiment-linked differences in "
                "acceptance remain modest and their confidence intervals include values "
                "close to zero, which is why the matched layer is interpreted as "
                "conservative validation rather than as the paper's strongest positive "
                "result."
            ),
            "tab:app:matchingeffects",
            "lrrrrr",
            float_fmt={"Matched contrast (ATT)": 3, "95% CI lower": 3, "95% CI upper": 3},
            size="small",
        ),
        latex_table_from_df(
            primary_balance_display,
            ["Covariate", "Absolute SMD before matching", "Absolute SMD after matching"],
            (
                "Standardized mean differences before and after matching for the primary "
                "borderline sample. The core balance target is $|SMD| < 0.10$ after "
                "matching. The table shows that the largest pre-match imbalance is in mean "
                "review length and that this imbalance contracts sharply after matching, "
                "along with the smaller imbalances in score, disagreement, confidence, and "
                "manuscript-side covariates."
            ),
            "tab:app:balance",
            "lrr",
            float_fmt={"Absolute SMD before matching": 3, "Absolute SMD after matching": 3},
        ),
        latex_table_from_df(
            overlap_display,
            [
                "Matching specification",
                "Analytic sample size",
                "Higher-sentiment papers",
                "Lower-sentiment papers",
                "Matched pairs",
                "Common-support lower bound",
                "Common-support upper bound",
                "Caliper",
                "Balance target met",
            ],
            (
                "Overlap summaries for the primary and sensitivity matched designs. The "
                "table reports analytic-sample size, the extent of common support, and the "
                "caliper used in each specification so that the reader can evaluate how "
                "much overlap is retained as the design window narrows or widens."
            ),
            "tab:app:overlap",
            "lrrrrrrrl",
            float_fmt={
                "Common-support lower bound": 3,
                "Common-support upper bound": 3,
                "Caliper": 3,
            },
            size="scriptsize",
            resize_to_width=True,
        ),
        latex_table_from_df(
            primary_counts_display
            if not primary_counts_display.empty
            else pd.DataFrame({"Year": YEARS, "Matched pairs": [0] * len(YEARS)}),
            ["Year", "Matched pairs"],
            (
                "Matched-pair counts by year for the primary borderline specification. The "
                "matched sample is distributed across all six conference years, indicating "
                "that the primary matched design is not driven by one unusually well-"
                "behaved cohort."
            ),
            "tab:app:matchedcounts",
            "rr",
        ),
        latex_table_from_df(
            topic_short_display,
            ["Topic cluster", "Submissions", "Representative keywords"],
            (
                "Deterministic keyword-based topic clusters used as topic fixed effects in "
                "the main models. The table is included for transparency because the paper "
                "does not claim official area-level stability; instead, these clusters "
                "serve as reproducible but noisy controls for topical composition."
            ),
            "tab:app:topics",
            "rrl",
        ),
        "\\FloatBarrier",
        "\\vspace{0.6em}",
        "\\section*{Appendix: Legacy Exploratory Materials}",
        (
            "The final appendix section preserves selected outputs from the earlier derived-"
            "table workflow. These materials remain useful as exploratory benchmarks and "
            "historical prototypes, but they are not the manuscript's primary evidentiary "
            "base."
        ),
        "",
        latex_figure_from_file(
            "appendix_figure_legacy_rq1.pdf",
            (
                "Legacy descriptive diagnostics from the earlier derived-table workflow, "
                "redrawn here in a cleaner format and retained as exploratory benchmarks "
                "for the expanded raw-archive rebuild. Their role is historical rather than "
                "inferential: they show that the archive-first package preserves the broad "
                "descriptive story while improving traceability and measurement "
                "transparency."
            ),
            "fig:app:legacyrq1",
        ),
        latex_figure_from_file(
            "appendix_figure_legacy_rq3.pdf",
            (
                "Legacy temporal and subgroup diagnostics from the earlier derived-table "
                "workflow, retained for comparison with the current archive-first "
                "analysis. Their role is archival rather than evidentiary: they show how "
                "the present manuscript reorganizes earlier stability patterns into a "
                "broader and more auditable design."
            ),
            "fig:app:legacyrq3",
        ),
        latex_table_from_df(
            legacy_multivariable_display
            if not legacy_multivariable_display.empty
            else pd.DataFrame(
                {"Predictor": [], "Odds ratio": [], "95% CI lower": [], "95% CI upper": [], "P-value": []}
            ),
            ["Predictor", "Odds ratio", "95% CI lower", "95% CI upper", "P-value"],
            (
                "Legacy multivariable logit benchmark from the earlier derived-table "
                "workflow. The table is retained only as a benchmark against the previous "
                "pipeline and should not be treated as the manuscript's primary source of "
                "inference."
            ),
            "tab:app:legacymultivariable",
            "lrrrr",
            float_fmt={"Odds ratio": 3, "95% CI lower": 3, "95% CI upper": 3, "P-value": 3},
            size="small",
        ),
        latex_table_from_df(
            legacy_meta_display
            if not legacy_meta_display.empty
            else pd.DataFrame({"Measure": [], "Pooled effect": [], "I-squared (%)": [], "Stability pattern": []}),
            ["Measure", "Pooled effect", "I-squared (%)", "Stability pattern"],
            (
                "Legacy stability summary from the earlier exploratory workflow. As with "
                "the previous legacy table, this summary is included to document "
                "continuity with the exploratory stage of the project rather than to "
                "substitute for the raw-archive analyses reported in the main text and "
                "earlier appendix sections."
            ),
            "tab:app:legacymeta",
            "lrrl",
            float_fmt={"Pooled effect": 3, "I-squared (%)": 3},
        ),
        "",
    ]
    APPENDIX_PATH.write_text("\n".join(sections))


def compute_summary_metrics(
    paper_df: pd.DataFrame,
    review_df: pd.DataFrame,
    observational: dict[str, pd.DataFrame],
    match_results: dict[str, object],
) -> dict[str, object]:
    by_year = (
        paper_df.groupby("year")
        .agg(papers=("forum", "count"), accept_rate=("accept", "mean"), mean_reviews=("num_reviews", "mean"))
        .reset_index()
    )
    return {
        "total_papers": int(len(paper_df)),
        "total_reviews": int(len(review_df)),
        "accept_rate": float(paper_df["accept"].mean()),
        "paper_sentiment_ame": float(
            observational["paper_ame"].set_index("term").loc["mean_sentiment", "ame"]
        ),
        "review_sentiment_ame": float(
            observational["review_ame"].set_index("term").loc["sentiment", "ame"]
        ),
        "matching_att": float(match_results["primary_diag"]["att"]),
        "matching_balance_pass": bool(match_results["primary_diag"]["balance_pass"]),
        "year_summary": by_year.to_dict(orient="records"),
    }


def write_outputs(
    review_df: pd.DataFrame,
    paper_df: pd.DataFrame,
    topic_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    observational: dict[str, pd.DataFrame],
    match_results: dict[str, object],
    prediction_df: pd.DataFrame,
    year_difference_df: pd.DataFrame,
    measurement_year_df: pd.DataFrame,
    bridge_df: pd.DataFrame,
    legacy_assets: dict[str, pd.DataFrame],
) -> None:
    review_df.to_csv(DERIVED_DIR / "review_level_canonical.csv", index=False)
    paper_df.to_csv(DERIVED_DIR / "paper_level_canonical.csv", index=False)
    topic_df.to_csv(DERIVED_DIR / "topic_cluster_summary.csv", index=False)
    audit_df.to_csv(DERIVED_DIR / "parsing_audit_by_forum.csv", index=False)
    measurement_year_df.to_csv(DERIVED_DIR / "measurement_year_summary.csv", index=False)
    bridge_df.to_csv(DERIVED_DIR / "score_bin_bridge.csv", index=False)

    for name, frame in observational.items():
        frame.to_csv(DERIVED_DIR / f"{name}.csv", index=False)

    for name, frame in legacy_assets.items():
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame.to_csv(DERIVED_DIR / f"{name}.csv", index=False)

    match_results["borderline_sample"].to_csv(DERIVED_DIR / "borderline_sample.csv", index=False)
    match_results["primary_work"].to_csv(DERIVED_DIR / "psm_primary_analytic_sample.csv", index=False)
    match_results["primary_pairs"].to_csv(DERIVED_DIR / "psm_primary_matched_pairs.csv", index=False)
    match_results["primary_balance"].to_csv(DERIVED_DIR / "psm_primary_balance.csv", index=False)
    match_results["primary_counts"].to_csv(DERIVED_DIR / "psm_primary_counts_by_year.csv", index=False)
    match_results["narrow_work"].to_csv(DERIVED_DIR / "psm_window_40_60_analytic_sample.csv", index=False)
    match_results["narrow_pairs"].to_csv(DERIVED_DIR / "psm_window_40_60_matched_pairs.csv", index=False)
    match_results["narrow_balance"].to_csv(DERIVED_DIR / "psm_window_40_60_balance.csv", index=False)
    match_results["wide_work"].to_csv(DERIVED_DIR / "psm_window_30_70_analytic_sample.csv", index=False)
    match_results["wide_pairs"].to_csv(DERIVED_DIR / "psm_window_30_70_matched_pairs.csv", index=False)
    match_results["wide_balance"].to_csv(DERIVED_DIR / "psm_window_30_70_balance.csv", index=False)
    match_results["robust_work"].to_csv(DERIVED_DIR / "psm_robust_analytic_sample.csv", index=False)
    match_results["robust_pairs"].to_csv(DERIVED_DIR / "psm_robust_matched_pairs.csv", index=False)
    match_results["robust_balance"].to_csv(DERIVED_DIR / "psm_robust_balance.csv", index=False)
    match_results["effects"].to_csv(DERIVED_DIR / "psm_effects.csv", index=False)
    match_results["overlap"].to_csv(DERIVED_DIR / "psm_overlap_summary.csv", index=False)
    prediction_df.to_csv(DERIVED_DIR / "cross_year_prediction_diagnostics.csv", index=False)
    year_difference_df.to_csv(DERIVED_DIR / "year_difference_effects.csv", index=False)

    summary = compute_summary_metrics(paper_df, review_df, observational, match_results)
    (DERIVED_DIR / "summary_metrics.json").write_text(json.dumps(summary, indent=2))
    write_numbers_tex(paper_df, observational, match_results, prediction_df, year_difference_df)
    write_appendix_tables(
        paper_df,
        review_df,
        topic_df,
        audit_df,
        observational,
        match_results,
        prediction_df,
        year_difference_df,
        measurement_year_df,
        bridge_df,
        legacy_assets,
    )


def refresh_outputs_from_tables(
    review_df: pd.DataFrame,
    paper_df: pd.DataFrame,
    topic_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    draw_legacy_figures: bool = True,
) -> None:
    observational = fit_observational_layers(review_df, paper_df)
    match_results = fit_matching_layer(paper_df)
    prediction_df = cross_year_prediction_diagnostics(paper_df)
    year_difference_df = compute_year_difference_effects(paper_df)
    measurement_year_df = compute_measurement_year_summary(paper_df, review_df)
    bridge_df = compute_score_bin_bridge(paper_df)
    legacy_assets = load_legacy_assets()
    write_outputs(
        review_df,
        paper_df,
        topic_df,
        audit_df,
        observational,
        match_results,
        prediction_df,
        year_difference_df,
        measurement_year_df,
        bridge_df,
        legacy_assets,
    )

    plot_appendix_measurement_summary(paper_df, review_df, measurement_year_df)
    plot_appendix_feature_correlations(paper_df)
    plot_appendix_score_bridge(bridge_df)
    plot_appendix_psm_overlap(match_results)
    plot_appendix_prediction_diagnostics(prediction_df)
    plot_appendix_topic_heterogeneity(observational, topic_df)
    if draw_legacy_figures:
        plot_appendix_legacy_rq1_montage(legacy_assets)
        plot_appendix_legacy_rq3_montage(legacy_assets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or refresh the causal paper package.")
    parser.add_argument(
        "--reuse-derived",
        action="store_true",
        help="Reuse existing canonical CSVs in paper/data/derived instead of rebuilding from the raw archive.",
    )
    args = parser.parse_args()

    ensure_dirs()
    setup_style()
    if args.reuse_derived:
        review_df, paper_df, topic_df, audit_df = load_existing_canonical_inputs()
    else:
        review_df, paper_df, topic_df, audit_df = build_canonical_tables()

    refresh_outputs_from_tables(review_df, paper_df, topic_df, audit_df)


if __name__ == "__main__":
    main()
