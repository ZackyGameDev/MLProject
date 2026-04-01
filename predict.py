"""
Predict future rating change for a Codeforces user.
====================================================
Usage:
    python predict.py <handle>
    python predict.py tourist
    python predict.py        (prompts for handle)

Fetches the user's data from the Codeforces API, computes the same
29 features used during training (from the most recent 90-day window),
and runs all 4 saved models to predict rating change over the next 30 days.
"""

import sys
import requests
import time
import numpy as np
import joblib
from collections import Counter, defaultdict
from math import log2

# ─── Config ───────────────────────────────────────────────────────────
DAY = 86400
FEATURE_DAYS = 90
FAMILIAR_TAG_THRESHOLD = 5
BASE = "https://codeforces.com/api"
SLEEP = 1.5


# ─── Codeforces API ──────────────────────────────────────────────────

def cf(endpoint, params=None):
    r = requests.get(BASE + endpoint, params=params)
    time.sleep(SLEEP)
    r.raise_for_status()
    data = r.json()
    if data["status"] != "OK":
        raise Exception(data.get("comment", "API error"))
    return data["result"]


# ─── Feature utility functions (same as cleaner.py) ──────────────────

def entropy(tags):
    if not tags:
        return 0.0
    c = Counter(tags)
    total = sum(c.values())
    return -sum((v / total) * log2(v / total) for v in c.values())


def top_k_concentration(tags, k=3):
    if not tags:
        return 0.0
    c = Counter(tags)
    total = sum(c.values())
    top_k = sum(v for _, v in c.most_common(k))
    return top_k / total


def longest_streak(sorted_days):
    if len(sorted_days) < 2:
        return len(sorted_days)
    best = cur = 1
    for i in range(1, len(sorted_days)):
        if sorted_days[i] == sorted_days[i - 1] + 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def weekly_submission_std(sub_times, f_start, f_end):
    if not sub_times:
        return 0.0
    weeks = defaultdict(int)
    total_weeks = max(1, int((f_end - f_start) / (7 * DAY)))
    for t in sub_times:
        week_idx = int((t - f_start) / (7 * DAY))
        weeks[week_idx] += 1
    counts = [weeks.get(w, 0) for w in range(total_weeks)]
    return float(np.std(counts)) if counts else 0.0


# ─── Feature extraction ─────────────────────────────────────────────

def extract_features(subs, contests):
    """
    Extract the 29 features from the most recent 90-day window
    that has contest activity.
    """
    if len(contests) < 2:
        raise ValueError("User has fewer than 2 rated contests. Not enough data.")

    subs_sorted = sorted(subs, key=lambda s: s["creationTimeSeconds"])
    contest_times = [c["ratingUpdateTimeSeconds"] for c in contests]

    # Use the most recent 90-day window ending at the last contest
    f_end = int(time.time())  # now
    f_start = f_end - FEATURE_DAYS * DAY

    window_subs = [s for s in subs if f_start <= s["creationTimeSeconds"] < f_end]
    window_contests = [c for c in contests if f_start <= c["ratingUpdateTimeSeconds"] < f_end]

    # If no activity in last 90 days, try the window ending at the last contest
    if not window_subs and not window_contests:
        f_end = max(contest_times)
        f_start = f_end - FEATURE_DAYS * DAY
        window_subs = [s for s in subs if f_start <= s["creationTimeSeconds"] < f_end]
        window_contests = [c for c in contests if f_start <= c["ratingUpdateTimeSeconds"] < f_end]

    if not window_subs:
        raise ValueError("No submissions found in the feature window.")

    # ── Base rating ──
    past_contests = [c for c in contests if c["ratingUpdateTimeSeconds"] < f_start]
    if past_contests:
        base_rating = past_contests[-1]["newRating"]
    else:
        base_rating = contests[0]["oldRating"]

    user_rating = window_contests[-1]["newRating"] if window_contests else base_rating

    # ── Historical context ──
    historical_subs = [s for s in subs_sorted if s["creationTimeSeconds"] < f_start]
    historical_tag_ac_counts = Counter()
    historical_tags_seen = set()
    for s in historical_subs:
        for tag in s["problem"].get("tags", []):
            historical_tags_seen.add(tag)
            if s["verdict"] == "OK":
                historical_tag_ac_counts[tag] += 1

    # ── Classify submissions ──
    contest_subs = [s for s in window_subs if s.get("author", {}).get("participantType") == "CONTESTANT"]
    practice_subs = [s for s in window_subs if s.get("author", {}).get("participantType") == "PRACTICE"]
    total_subs = len(window_subs)
    ac_subs = [s for s in window_subs if s["verdict"] == "OK"]
    ac_practice = [s for s in practice_subs if s["verdict"] == "OK"]

    # ═══ UPSOLVING ═══
    upsolve_count = len(ac_practice)
    upsolve_ratio = upsolve_count / max(1, len(practice_subs)) if practice_subs else 0.0
    upsolve_ratings = [s["problem"].get("rating", 0) for s in ac_practice if s["problem"].get("rating")]
    upsolve_difficulty_delta = (np.mean(upsolve_ratings) - user_rating) if upsolve_ratings else 0.0
    contest_to_practice_ratio = len(practice_subs) / max(1, len(contest_subs)) if contest_subs else 0.0

    # ═══ CONSISTENCY ═══
    ac_ratio = len(ac_subs) / max(1, total_subs)
    days = sorted(set(s["creationTimeSeconds"] // DAY for s in window_subs))
    active_days = len(days)

    if len(days) >= 2:
        gaps = np.diff(days)
        avg_gap_days = float(np.mean(gaps))
        max_gap = float(np.max(gaps))
    else:
        avg_gap_days = float(FEATURE_DAYS)
        max_gap = float(FEATURE_DAYS)

    sub_times_window = [s["creationTimeSeconds"] for s in window_subs]
    weekly_con = weekly_submission_std(sub_times_window, f_start, f_end)
    streak = longest_streak(days)

    window_tags = []
    familiar_count = 0
    new_count = 0
    for s in window_subs:
        for tag in s["problem"].get("tags", []):
            window_tags.append(tag)
            if historical_tag_ac_counts[tag] >= FAMILIAR_TAG_THRESHOLD:
                familiar_count += 1
            else:
                new_count += 1
    speed_vs_growth = familiar_count / max(1, new_count)

    # ═══ IQ ═══
    rating_delta = (
        window_contests[-1]["newRating"] - window_contests[0]["oldRating"]
        if len(window_contests) >= 2 else 0
    )

    if len(window_contests) >= 2:
        per_contest_deltas = [c["newRating"] - c["oldRating"] for c in window_contests]
        rating_volatility = float(np.std(per_contest_deltas))
    else:
        rating_volatility = 0.0

    if len(window_contests) >= 2:
        times = np.array([c["ratingUpdateTimeSeconds"] for c in window_contests], dtype=float)
        ratings = np.array([c["newRating"] for c in window_contests], dtype=float)
        times_days = (times - times[0]) / DAY
        if times_days[-1] > 0:
            rating_trend_slope = float(np.polyfit(times_days, ratings, 1)[0])
        else:
            rating_trend_slope = 0.0
    else:
        rating_trend_slope = 0.0

    if window_contests:
        ranks = [c.get("rank", 0) for c in window_contests]
        contest_rank_percentile_avg = float(np.mean(ranks))
    else:
        contest_rank_percentile_avg = 0.0

    contest_ac = [s for s in contest_subs if s["verdict"] == "OK"]
    if contest_ac:
        solve_times = [s.get("relativeTimeSeconds", 0) for s in contest_ac]
        solve_speed_avg = float(np.mean(solve_times)) / 60.0
    else:
        solve_speed_avg = 0.0

    ac_ratings = [s["problem"].get("rating", 0) for s in ac_subs if s["problem"].get("rating")]
    difficulty_ceiling = max(ac_ratings) if ac_ratings else 0

    prob_ratings = [s["problem"].get("rating") for s in window_subs if s["problem"].get("rating")]
    problem_rating_vs_user = (np.mean(prob_ratings) / max(800, user_rating)) if prob_ratings else 0.0

    # ═══ TOPIC DIVERSITY ═══
    tag_ent = entropy(window_tags)
    unique_tags = len(set(window_tags))
    window_tags_set = set(window_tags)
    new_tags_explored = len(window_tags_set - historical_tags_seen)
    tag_conc_top3 = top_k_concentration(window_tags)

    # ═══ LEARNING EFFICIENCY ═══
    attempts = Counter(
        (s["problem"].get("name", ""), s["problem"].get("index", ""))
        for s in window_subs
    )
    attempts_per_ac = float(np.mean(list(attempts.values()))) if attempts else 0.0

    problem_first_verdict = {}
    for s in sorted(window_subs, key=lambda x: x["creationTimeSeconds"]):
        pid = (s["problem"].get("contestId", ""), s["problem"].get("index", ""))
        if pid not in problem_first_verdict:
            problem_first_verdict[pid] = s["verdict"]
    if problem_first_verdict:
        first_ac = sum(1 for v in problem_first_verdict.values() if v == "OK")
        first_attempt_ac_rate = first_ac / len(problem_first_verdict)
    else:
        first_attempt_ac_rate = 0.0

    problem_events = defaultdict(list)
    for s in sorted(window_subs, key=lambda x: x["creationTimeSeconds"]):
        pid = (s["problem"].get("contestId", ""), s["problem"].get("index", ""))
        problem_events[pid].append((s["creationTimeSeconds"], s["verdict"]))
    debug_times = []
    for pid, events in problem_events.items():
        first_wa = None
        last_ac = None
        for ts, verdict in events:
            if verdict != "OK" and first_wa is None:
                first_wa = ts
            if verdict == "OK":
                last_ac = ts
        if first_wa is not None and last_ac is not None and last_ac > first_wa:
            debug_times.append((last_ac - first_wa) / 60.0)
    avg_debug_time = float(np.mean(debug_times)) if debug_times else 0.0

    # ═══ CONTEST BEHAVIOR ═══
    contest_count = len(window_contests)

    if contest_subs:
        problems_per_contest = defaultdict(set)
        for s in contest_subs:
            if s["verdict"] == "OK":
                cid = s.get("contestId", s["problem"].get("contestId", ""))
                pid = (cid, s["problem"].get("index", ""))
                problems_per_contest[cid].add(pid)
        contest_problems_solved_avg = float(np.mean([len(v) for v in problems_per_contest.values()])) if problems_per_contest else 0.0
    else:
        contest_problems_solved_avg = 0.0

    # Return as dict
    return {
        "upsolve_count": upsolve_count,
        "upsolve_ratio": upsolve_ratio,
        "upsolve_difficulty_delta": upsolve_difficulty_delta,
        "contest_to_practice_ratio": contest_to_practice_ratio,
        "subs_90d": total_subs,
        "ac_ratio": ac_ratio,
        "active_days": active_days,
        "avg_gap_days": avg_gap_days,
        "max_gap_days": max_gap,
        "weekly_consistency": weekly_con,
        "streak_max": streak,
        "speed_vs_growth_ratio": speed_vs_growth,
        "base_rating": base_rating,
        "rating_delta_90d": rating_delta,
        "rating_volatility": rating_volatility,
        "rating_trend_slope": rating_trend_slope,
        "contest_rank_percentile_avg": contest_rank_percentile_avg,
        "solve_speed_avg": solve_speed_avg,
        "difficulty_ceiling": difficulty_ceiling,
        "problem_rating_vs_user_rating": problem_rating_vs_user,
        "tag_entropy": tag_ent,
        "unique_tags_count": unique_tags,
        "new_tags_explored": new_tags_explored,
        "tag_concentration_top3": tag_conc_top3,
        "attempts_per_ac": attempts_per_ac,
        "first_attempt_ac_rate": first_attempt_ac_rate,
        "avg_debug_time": avg_debug_time,
        "contest_count": contest_count,
        "contest_problems_solved_avg": contest_problems_solved_avg,
    }


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    # Get handle
    if len(sys.argv) > 1:
        handle = sys.argv[1]
    else:
        handle = input("Enter Codeforces handle: ").strip()

    if not handle:
        print("No handle provided.")
        return

    print(f"\nFetching data for '{handle}' from Codeforces API...")

    try:
        info = cf("/user.info", {"handles": handle})[0]
        rating_history = cf("/user.rating", {"handle": handle})
        submissions = cf("/user.status", {"handle": handle, "count": 10000})
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    current_rating = info.get("rating", "Unrated")
    max_rating = info.get("maxRating", "N/A")
    rank = info.get("rank", "N/A")

    print(f"  Current rating: {current_rating}  |  Max: {max_rating}  |  Rank: {rank}")
    print(f"  Contests: {len(rating_history)}  |  Submissions: {len(submissions)}")

    # Extract features
    print("\nComputing features...")
    try:
        features = extract_features(submissions, rating_history)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Load models and feature order
    print("Loading models...")
    feature_cols = joblib.load("models/feature_columns.joblib")
    scaler = joblib.load("models/scaler.joblib")

    # Build feature vector in the correct order
    X = np.array([[features[col] for col in feature_cols]])

    # Print feature summary
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)

    factor_groups = {
        "Upsolving": ["upsolve_count", "upsolve_ratio", "upsolve_difficulty_delta", "contest_to_practice_ratio"],
        "Consistency": ["subs_90d", "ac_ratio", "active_days", "avg_gap_days", "max_gap_days", "weekly_consistency", "streak_max", "speed_vs_growth_ratio"],
        "Cognitive": ["base_rating", "rating_delta_90d", "rating_volatility", "rating_trend_slope", "contest_rank_percentile_avg", "solve_speed_avg", "difficulty_ceiling", "problem_rating_vs_user_rating"],
        "Diversity": ["tag_entropy", "unique_tags_count", "new_tags_explored", "tag_concentration_top3"],
        "Efficiency": ["attempts_per_ac", "first_attempt_ac_rate", "avg_debug_time"],
        "Contest": ["contest_count", "contest_problems_solved_avg"],
    }

    for group, cols in factor_groups.items():
        print(f"\n  [{group}]")
        for col in cols:
            val = features[col]
            if isinstance(val, float):
                print(f"    {col:>35s}: {val:.4f}")
            else:
                print(f"    {col:>35s}: {val}")

    # Run predictions
    print("\n" + "=" * 60)
    print("PREDICTIONS (rating change over next 30 days)")
    print("=" * 60)

    model_files = {
        "Ridge": ("models/ridge.joblib", True),
        "SVR (RBF)": ("models/svr_rbf.joblib", True),
        "Random Forest": ("models/random_forest.joblib", False),
        "XGBoost": ("models/xgboost.joblib", False),
    }

    predictions = {}
    for name, (path, needs_scaling) in model_files.items():
        try:
            model = joblib.load(path)
            X_input = scaler.transform(X) if needs_scaling else X
            pred = model.predict(X_input)[0]
            predictions[name] = pred
            print(f"  {name:<20s}: {pred:>+8.1f} rating points")
        except Exception as e:
            print(f"  {name:<20s}: Error - {e}")

    if predictions:
        avg_pred = np.mean(list(predictions.values()))
        print(f"\n  {'Ensemble avg':<20s}: {avg_pred:>+8.1f} rating points")

        print("\n" + "-" * 60)
        if avg_pred > 50:
            print("  Outlook: GROWTH -- This user's behavior suggests improvement")
        elif avg_pred > -50:
            print("  Outlook: STABLE -- This user's behavior suggests stagnation")
        else:
            print("  Outlook: DECLINE -- This user's behavior suggests regression")
        print("-" * 60)


if __name__ == "__main__":
    main()
