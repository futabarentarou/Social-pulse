"""
dashboard.py
------------
Reporting / visualization stage of Social Pulse.

Produces the kind of charts a Social Media Analyst hands to the account,
strategy, and creative teams:

  1. Sentiment breakdown (how the audience feels)
  2. Conversation volume over time (when the campaign spiked)
  3. Top posts by Engagement Resonance Score (what resonated)
  4. Average engagement by subreddit / community (where to focus)
  5. Engagement by posting hour (when to publish)

All figures are saved to outputs/ as PNGs for use in slide decks / reports.
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
OUT_DIR = "outputs"


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def sentiment_breakdown(df):
    counts = df["sentiment_label"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    palette = {"positive": "#2e7d32", "neutral": "#9e9e9e", "negative": "#c62828"}
    colors = [palette.get(k, "#607d8b") for k in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Audience sentiment breakdown")
    ax.set_ylabel("Number of posts")
    _save(fig, "01_sentiment_breakdown.png")


def conversation_over_time(df):
    ts = df.copy()
    ts["date"] = pd.to_datetime(ts["date"])
    daily = ts.groupby("date").size()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily.values, color="#1565c0")
    ax.fill_between(daily.index, daily.values, alpha=0.2, color="#1565c0")
    ax.set_title("Conversation volume over time")
    ax.set_ylabel("Posts per day")
    ax.set_xlabel("Date")
    fig.autofmt_xdate()
    _save(fig, "02_conversation_over_time.png")


def top_posts(df, n=10):
    top = df.sort_values("engagement_resonance_score", ascending=False).head(n)
    labels = top["title"].str.slice(0, 55) + "..."
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels[::-1], top["engagement_resonance_score"][::-1], color="#6a1b9a")
    ax.set_title(f"Top {n} posts by Engagement Resonance Score")
    ax.set_xlabel("Engagement Resonance Score (0-100)")
    _save(fig, "03_top_posts.png")


def engagement_by_community(df):
    by_sub = (
        df.groupby("subreddit")["engagement_resonance_score"]
        .mean()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(by_sub.index, by_sub.values, color="#00897b")
    ax.set_title("Average engagement by community")
    ax.set_ylabel("Mean Engagement Resonance Score")
    plt.xticks(rotation=30, ha="right")
    _save(fig, "04_engagement_by_community.png")


def engagement_by_hour(df):
    by_hour = df.groupby("hour_utc")["engagement_resonance_score"].mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(by_hour.index, by_hour.values, marker="o", color="#ef6c00")
    ax.set_title("Average engagement by posting hour (UTC)")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Mean Engagement Resonance Score")
    ax.set_xticks(range(0, 24, 2))
    _save(fig, "05_engagement_by_hour.png")


def build_all(df):
    print("Building dashboard figures...")
    sentiment_breakdown(df)
    conversation_over_time(df)
    top_posts(df)
    engagement_by_community(df)
    engagement_by_hour(df)
    print("Done. See the outputs/ folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dashboard figures.")
    parser.add_argument("--in", dest="inp", default="data/processed/posts_features.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.inp)
    build_all(df)
