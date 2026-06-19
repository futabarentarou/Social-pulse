"""
features.py
-----------
Social-listening and feature-engineering stage of Social Pulse.

Takes the raw Reddit posts collected by collect_reddit.py and turns each post
into a row of measurable social-media signals that a Social Media Analyst would
actually report on:

  - sentiment (VADER, tuned for social text)
  - text characteristics (length, question/exclamation, caps ratio)
  - timing (hour of day, weekend flag)
  - a custom Engagement Resonance Score (ERS) that blends reach, reaction
    intensity, and conversation depth into one interpretable number.

The ERS is the analogue of the "resonance index" pattern: a transparent,
hand-built metric that translates raw platform data into a single signal the
account/strategy teams can rank content by.
"""

import re

import numpy as np
import pandas as pd

# Prefer VADER (tuned for social-media text). If it isn't installed, fall back
# to a small built-in lexicon scorer so the pipeline still runs everywhere.
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _ANALYZER = SentimentIntensityAnalyzer()

    def _polarity(text: str) -> dict:
        return _ANALYZER.polarity_scores(text)

except ImportError:  # lightweight fallback
    _POS_WORDS = {
        "love", "loved", "great", "amazing", "best", "stunning", "worth",
        "chills", "nailed", "comeback", "incredible", "perfect", "beautiful",
        "favorite", "excellent", "brilliant", "masterpiece", "hooked",
    }
    _NEG_WORDS = {
        "disappointed", "lazy", "ruined", "wasted", "overhyped", "cheap",
        "bad", "worst", "boring", "awful", "terrible", "hate", "cashgrab",
        "cash", "grab", "fell", "couldn't", "couldnt", "weak", "mess",
    }

    def _polarity(text: str) -> dict:
        words = re.findall(r"[a-z']+", text.lower())
        if not words:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        pos = sum(1 for w in words if w in _POS_WORDS)
        neg = sum(1 for w in words if w in _NEG_WORDS)
        total = len(words)
        pos_r, neg_r = pos / total, neg / total
        compound = float(np.tanh(3 * (pos_r - neg_r)))
        neu_r = max(0.0, 1.0 - pos_r - neg_r)
        return {"compound": compound, "pos": pos_r, "neg": neg_r, "neu": neu_r}


def _full_text(row) -> str:
    title = str(row.get("title", "") or "")
    body = str(row.get("selftext", "") or "")
    return (title + " " + body).strip()


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text = df.apply(_full_text, axis=1)

    df["char_count"] = text.str.len()
    df["word_count"] = text.str.split().apply(len)
    df["has_question"] = text.str.contains(r"\?", regex=True).astype(int)
    df["has_exclamation"] = text.str.contains(r"!", regex=True).astype(int)

    def caps_ratio(s):
        letters = re.findall(r"[A-Za-z]", s)
        if not letters:
            return 0.0
        caps = sum(1 for c in letters if c.isupper())
        return caps / len(letters)

    df["caps_ratio"] = text.apply(caps_ratio)
    return df


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text = df.apply(_full_text, axis=1)
    scores = text.apply(_polarity)
    df["sentiment_compound"] = scores.apply(lambda d: d["compound"])
    df["sentiment_pos"] = scores.apply(lambda d: d["pos"])
    df["sentiment_neg"] = scores.apply(lambda d: d["neg"])
    df["sentiment_neu"] = scores.apply(lambda d: d["neu"])

    def label(c):
        if c >= 0.05:
            return "positive"
        if c <= -0.05:
            return "negative"
        return "neutral"

    df["sentiment_label"] = df["sentiment_compound"].apply(label)
    return df


def add_timing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["created_utc"], utc=True, errors="coerce")
    df["hour_utc"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek          # 0 = Monday
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["date"] = ts.dt.date
    return df


def add_engagement_resonance_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engagement Resonance Score (ERS)
    --------------------------------
    A transparent, interpretable index built from three social signals:

      reach_signal     = log(score + 1)             -> how far it traveled
      reaction_signal  = upvote_ratio               -> how positively it landed
      conversation     = log(num_comments + 1)      -> how much talk it sparked

    Each is min-max normalized to 0-1 across the dataset, then combined with
    weights that emphasize conversation depth (what social teams care about for
    organic campaigns). The result is scaled to 0-100 for readability.
    """
    df = df.copy()

    reach = np.log1p(df["score"].clip(lower=0))
    reaction = df["upvote_ratio"].fillna(0.5)
    conversation = np.log1p(df["num_comments"].clip(lower=0))

    def norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    reach_n = norm(reach)
    reaction_n = norm(reaction)
    conversation_n = norm(conversation)

    df["reach_signal"] = reach_n
    df["reaction_signal"] = reaction_n
    df["conversation_signal"] = conversation_n

    df["engagement_resonance_score"] = (
        0.35 * reach_n + 0.25 * reaction_n + 0.40 * conversation_n
    ) * 100

    # Binary target for the modeling stage: top 25% of posts by ERS = "high engagement".
    threshold = df["engagement_resonance_score"].quantile(0.75)
    df["high_engagement"] = (
        df["engagement_resonance_score"] >= threshold
    ).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline."""
    df = add_text_features(df)
    df = add_sentiment(df)
    df = add_timing(df)
    df = add_engagement_resonance_score(df)
    return df


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Build social features.")
    parser.add_argument("--in", dest="inp", default="data/raw/reddit_posts.csv")
    parser.add_argument("--out", default="data/processed/posts_features.csv")
    args = parser.parse_args()

    raw = pd.read_csv(args.inp)
    feats = build_features(raw)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    feats.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Saved {len(feats)} rows with features to {args.out}")
