"""
generate_demo_data.py
---------------------
Creates a realistic SYNTHETIC dataset of Reddit-style posts about an
entertainment campaign, so the full pipeline can be run and demonstrated
WITHOUT Reddit credentials.

This is useful for:
  - reviewers who want to run the repo immediately,
  - reproducible screenshots / the demo dashboard,
  - CI or quick local testing.

The structure matches exactly what collect_reddit.py produces, so the rest of
the pipeline (features -> model -> dashboard) is identical on real or demo data.
"""

import argparse
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

SUBREDDITS = ["television", "movies", "netflix", "anime", "entertainment"]

POSITIVE_SNIPPETS = [
    "absolutely loved the finale, best show this year",
    "the cinematography is stunning, can't stop thinking about it",
    "this trailer gave me chills, instant watchlist",
    "the cast nailed every scene, what a comeback",
    "binged the whole season in one night, worth every minute",
]
NEGATIVE_SNIPPETS = [
    "honestly disappointed, the pacing fell apart halfway",
    "felt like a cash grab, the writing was lazy",
    "the ending ruined it for me, such wasted potential",
    "way overhyped, couldn't finish the second episode",
    "the cgi looked cheap and took me out of it",
]
NEUTRAL_SNIPPETS = [
    "anyone know the release date for the next part",
    "where can I stream this outside the US",
    "is this based on the book or an original story",
    "how many episodes are in the first season",
    "what time does the new episode drop",
]


def generate(n: int, query: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    base_time = datetime.now(tz=timezone.utc) - timedelta(days=180)

    for i in range(n):
        bucket = rng.choice(
            ["pos", "neg", "neu"], p=[0.45, 0.25, 0.30]
        )
        if bucket == "pos":
            snippet = rng.choice(POSITIVE_SNIPPETS)
            base_score = rng.gamma(3.0, 120)
            base_comments = rng.gamma(2.5, 25)
            upvote_ratio = rng.uniform(0.80, 0.98)
        elif bucket == "neg":
            snippet = rng.choice(NEGATIVE_SNIPPETS)
            base_score = rng.gamma(2.0, 60)
            base_comments = rng.gamma(2.8, 30)  # negativity drives comments
            upvote_ratio = rng.uniform(0.45, 0.75)
        else:
            snippet = rng.choice(NEUTRAL_SNIPPETS)
            base_score = rng.gamma(1.5, 40)
            base_comments = rng.gamma(1.5, 10)
            upvote_ratio = rng.uniform(0.70, 0.92)

        title = f"{query}: {snippet}"
        created = base_time + timedelta(
            hours=int(rng.uniform(0, 180 * 24))
        )

        rows.append(
            {
                "post_id": f"demo_{i:05d}",
                "subreddit": rng.choice(SUBREDDITS),
                "title": title,
                "selftext": "" if rng.random() < 0.6 else snippet,
                "score": int(max(0, base_score)),
                "upvote_ratio": round(float(upvote_ratio), 2),
                "num_comments": int(max(0, base_comments)),
                "created_utc": created.isoformat(),
                "author": f"user_{rng.integers(1, 5000)}",
                "url": "https://example.com",
                "is_self": bool(rng.random() < 0.4),
                "over_18": False,
                "permalink": f"https://reddit.com/demo/{i}",
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic demo data.")
    parser.add_argument("--n", type=int, default=1200, help="Number of posts.")
    parser.add_argument("--query", default="The Studio Show", help="Campaign name.")
    parser.add_argument("--out", default="data/raw/reddit_posts.csv")
    args = parser.parse_args()

    df = generate(args.n, args.query)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Generated {len(df)} synthetic posts -> {args.out}")


if __name__ == "__main__":
    main()
