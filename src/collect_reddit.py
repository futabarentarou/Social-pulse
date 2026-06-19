"""
collect_reddit.py
-----------------
Collects social media conversation data from Reddit about an entertainment
title (movie / TV show / streaming release) and saves it as a raw CSV.

This is the data-collection stage of the Social Pulse project. It uses the
FREE Reddit Data API tier (100 queries/min, non-commercial use), accessed
through the PRAW library.

HOW TO GET FREE CREDENTIALS (10 minutes, no card required):
  1. Go to https://www.reddit.com/prefs/apps
  2. Click "create another app" at the bottom.
  3. Choose type "script".
  4. Name: social-pulse  |  redirect uri: http://localhost:8080
  5. After creating, copy the client_id (under the app name) and the secret.
  6. Put them in a .env file (see .env.example).

USAGE:
  python src/collect_reddit.py --query "the studio show" --subreddits television movies --limit 500
"""

import argparse
import os
import time
from datetime import datetime, timezone

import pandas as pd
import praw
from dotenv import load_dotenv


def get_reddit_client() -> praw.Reddit:
    """Build an authenticated, read-only PRAW client from environment vars."""
    load_dotenv()
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "social-pulse/0.1 by u/your_username")

    if not client_id or not client_secret:
        raise RuntimeError(
            "Missing Reddit credentials. Copy .env.example to .env and fill "
            "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET."
        )

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )
    reddit.read_only = True
    return reddit


def collect_posts(reddit, query, subreddits, limit_per_sub, time_filter="year"):
    """Search each subreddit for the query and return a list of post records."""
    rows = []
    for sub in subreddits:
        print(f"  Searching r/{sub} for '{query}' ...")
        subreddit = reddit.subreddit(sub)
        try:
            results = subreddit.search(
                query, sort="relevance", time_filter=time_filter, limit=limit_per_sub
            )
            for post in results:
                rows.append(
                    {
                        "post_id": post.id,
                        "subreddit": sub,
                        "title": post.title,
                        "selftext": post.selftext,
                        "score": post.score,                 # net upvotes
                        "upvote_ratio": post.upvote_ratio,    # share of upvotes
                        "num_comments": post.num_comments,    # engagement
                        "created_utc": datetime.fromtimestamp(
                            post.created_utc, tz=timezone.utc
                        ).isoformat(),
                        "author": str(post.author),
                        "url": post.url,
                        "is_self": post.is_self,
                        "over_18": post.over_18,
                        "permalink": f"https://reddit.com{post.permalink}",
                    }
                )
            # Stay polite well within the 100 QPM free-tier limit.
            time.sleep(1)
        except Exception as exc:  # network / rate-limit / private sub
            print(f"    [warn] could not read r/{sub}: {exc}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Collect Reddit posts for a campaign.")
    parser.add_argument(
        "--query", required=True, help="Entertainment title / campaign keyword."
    )
    parser.add_argument(
        "--subreddits",
        nargs="+",
        default=["television", "movies", "netflix", "anime", "entertainment"],
        help="Subreddits to search.",
    )
    parser.add_argument(
        "--limit", type=int, default=500, help="Max posts per subreddit (<=1000)."
    )
    parser.add_argument(
        "--time", default="year", help="Time filter: hour/day/week/month/year/all."
    )
    parser.add_argument(
        "--out", default="data/raw/reddit_posts.csv", help="Output CSV path."
    )
    args = parser.parse_args()

    reddit = get_reddit_client()
    print(f"Collecting posts for campaign keyword: '{args.query}'")
    rows = collect_posts(reddit, args.query, args.subreddits, args.limit, args.time)

    df = pd.DataFrame(rows).drop_duplicates(subset="post_id")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} unique posts to {args.out}")


if __name__ == "__main__":
    main()
