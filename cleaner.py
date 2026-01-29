import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from math import log2

FEATURE_DAYS = 90
LABEL_DAYS = 30
STEP_DAYS = 30
DAY = 86400

rows = []

def entropy(tags):
    if not tags:
        return 0
    c = Counter(tags)
    total = sum(c.values())
    return -sum((v / total) * log2(v / total) for v in c.values())

print("Streaming raw jsonl...")

with open("codeforces_dataset.jsonl", "r", encoding="utf8") as f:
    for line in tqdm(f):

        user = json.loads(line)

        handle = user["handle"]
        subs = user["submissions"]
        contests = user["rating_history"]

        if len(contests) < 2:
            continue

        sub_times = np.array([s["creationTimeSeconds"] for s in subs])
        contest_times = np.array([c["ratingUpdateTimeSeconds"] for c in contests])

        start = min(
            sub_times.min() if len(sub_times) else contest_times.min(),
            contest_times.min()
        )

        end = contest_times.max()
        t = start

        while t + (FEATURE_DAYS + LABEL_DAYS) * DAY < end:

            f_start = t
            f_end = t + FEATURE_DAYS * DAY
            l_end = f_end + LABEL_DAYS * DAY

            window_subs = [s for s in subs if f_start <= s["creationTimeSeconds"] < f_end]
            window_contests = [c for c in contests if f_start <= c["ratingUpdateTimeSeconds"] < f_end]
            label_contests = [c for c in contests if f_end <= c["ratingUpdateTimeSeconds"] < l_end]

            if not label_contests:
                t += STEP_DAYS * DAY
                continue

            total_subs = len(window_subs)
            ac = [s for s in window_subs if s["verdict"] == "OK"]

            prob_ratings = [
                s["problem"].get("rating")
                for s in window_subs
                if s["problem"].get("rating")
            ]

            tags = []
            for s in window_subs:
                tags += s["problem"].get("tags", [])

            attempts = Counter(
                (
                    s["problem"].get("name", ""),
                    s["problem"].get("index", "")
                )
                for s in window_subs
            )

            attempts_per_ac = np.mean(list(attempts.values())) if attempts else 0

            days = sorted(set(s["creationTimeSeconds"] // DAY for s in window_subs))
            max_gap = max(np.diff(days)) if len(days) >= 2 else FEATURE_DAYS

            if len(window_contests) >= 2:
                rating_delta = window_contests[-1]["newRating"] - window_contests[0]["oldRating"]
            else:
                rating_delta = 0

            future_delta = label_contests[-1]["newRating"] - label_contests[0]["oldRating"]

            rows.append({
                "handle": handle,
                "subs_90d": total_subs,
                "ac_ratio": len(ac) / max(1, total_subs),
                "avg_problem_rating": np.mean(prob_ratings) if prob_ratings else 0,
                "tag_entropy": entropy(tags),
                "attempts_per_ac": attempts_per_ac,
                "contest_count": len(window_contests),
                "rating_delta_90d": rating_delta,
                "max_gap_days": max_gap,
                "future_rating_delta_30d": future_delta
            })

            t += STEP_DAYS * DAY

print("Building dataframe...")

df = pd.DataFrame(rows)
df.to_csv("cf_ml_dataset.csv", index=False)

print("Saved cf_ml_dataset.csv")
