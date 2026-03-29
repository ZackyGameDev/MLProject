import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict
from math import log2

FEATURE_DAYS = 90
LABEL_DAYS = 30
STEP_DAYS = 30
DAY = 86400
FAMILIAR_TAG_THRESHOLD = 5  # min past ACs to consider a tag "familiar"

rows = []


# ─── Utility functions ───────────────────────────────────────────────

def entropy(tags):
    """Shannon entropy over tag frequency distribution."""
    if not tags:
        return 0.0
    c = Counter(tags)
    total = sum(c.values())
    return -sum((v / total) * log2(v / total) for v in c.values())


def top_k_concentration(tags, k=3):
    """Fraction of total tags accounted for by the top-k most common."""
    if not tags:
        return 0.0
    c = Counter(tags)
    total = sum(c.values())
    top_k = sum(v for _, v in c.most_common(k))
    return top_k / total


def longest_streak(sorted_days):
    """Longest consecutive-day streak from a sorted list of unique day indices."""
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
    """Std deviation of per-week submission counts (lower = more consistent)."""
    if not sub_times:
        return 0.0
    weeks = defaultdict(int)
    total_weeks = max(1, int((f_end - f_start) / (7 * DAY)))
    for t in sub_times:
        week_idx = int((t - f_start) / (7 * DAY))
        weeks[week_idx] += 1
    counts = [weeks.get(w, 0) for w in range(total_weeks)]
    return float(np.std(counts)) if counts else 0.0


# ─── Main processing ─────────────────────────────────────────────────

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

        # Pre-compute cumulative tag history for "new tags" / "familiar tags"
        # Sort all submissions chronologically
        subs_sorted = sorted(subs, key=lambda s: s["creationTimeSeconds"])

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

            # ═══════════════════════════════════════════════════
            # BASE RATING
            # ═══════════════════════════════════════════════════
            past_contests = [c for c in contests if c["ratingUpdateTimeSeconds"] < f_start]
            if past_contests:
                base_rating = past_contests[-1]["newRating"]
            else:
                base_rating = contests[0]["oldRating"]

            user_rating = window_contests[-1]["newRating"] if window_contests else base_rating

            # ═══════════════════════════════════════════════════
            # HISTORICAL CONTEXT (before this window)
            # ═══════════════════════════════════════════════════
            historical_subs = [s for s in subs_sorted if s["creationTimeSeconds"] < f_start]

            # Tags the user has seen before this window
            historical_tag_ac_counts = Counter()
            historical_tags_seen = set()
            for s in historical_subs:
                for tag in s["problem"].get("tags", []):
                    historical_tags_seen.add(tag)
                    if s["verdict"] == "OK":
                        historical_tag_ac_counts[tag] += 1

            # ═══════════════════════════════════════════════════
            # CLASSIFY SUBMISSIONS
            # ═══════════════════════════════════════════════════
            contest_subs = [s for s in window_subs if s.get("author", {}).get("participantType") == "CONTESTANT"]
            practice_subs = [s for s in window_subs if s.get("author", {}).get("participantType") == "PRACTICE"]

            total_subs = len(window_subs)
            ac_subs = [s for s in window_subs if s["verdict"] == "OK"]
            ac_practice = [s for s in practice_subs if s["verdict"] == "OK"]

            # ═══════════════════════════════════════════════════
            # 1. UPSOLVING FEATURES
            # ═══════════════════════════════════════════════════

            # Upsolve count: AC practice submissions
            upsolve_count = len(ac_practice)

            # Upsolve ratio: fraction of practice submissions that are AC (upsolving intensity)
            upsolve_ratio = upsolve_count / max(1, len(practice_subs)) if practice_subs else 0.0

            # Upsolve difficulty delta: avg rating of upsolved problems - user rating
            upsolve_ratings = [s["problem"].get("rating", 0) for s in ac_practice if s["problem"].get("rating")]
            upsolve_difficulty_delta = (np.mean(upsolve_ratings) - user_rating) if upsolve_ratings else 0.0

            # Contest to practice ratio
            contest_to_practice_ratio = len(practice_subs) / max(1, len(contest_subs)) if contest_subs else 0.0

            # ═══════════════════════════════════════════════════
            # 2. CONSISTENCY FEATURES
            # ═══════════════════════════════════════════════════

            ac_ratio = len(ac_subs) / max(1, total_subs)

            # Active days & gaps
            days = sorted(set(s["creationTimeSeconds"] // DAY for s in window_subs))
            active_days = len(days)

            if len(days) >= 2:
                gaps = np.diff(days)
                avg_gap_days = float(np.mean(gaps))
                max_gap = float(np.max(gaps))
            else:
                avg_gap_days = float(FEATURE_DAYS)
                max_gap = float(FEATURE_DAYS)

            # Weekly consistency (std of weekly sub counts)
            sub_times_window = [s["creationTimeSeconds"] for s in window_subs]
            weekly_con = weekly_submission_std(sub_times_window, f_start, f_end)

            # Longest streak
            streak = longest_streak(days)

            # Speed vs growth ratio
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

            # ═══════════════════════════════════════════════════
            # 3. IQ / COGNITIVE ABILITY FEATURES
            # ═══════════════════════════════════════════════════

            # Rating delta
            rating_delta = (
                window_contests[-1]["newRating"] - window_contests[0]["oldRating"]
                if len(window_contests) >= 2 else 0
            )

            # Rating volatility (std of per-contest changes)
            if len(window_contests) >= 2:
                per_contest_deltas = [c["newRating"] - c["oldRating"] for c in window_contests]
                rating_volatility = float(np.std(per_contest_deltas))
            else:
                rating_volatility = 0.0

            # Rating trend slope (linear regression of rating over time)
            if len(window_contests) >= 2:
                times = np.array([c["ratingUpdateTimeSeconds"] for c in window_contests], dtype=float)
                ratings = np.array([c["newRating"] for c in window_contests], dtype=float)
                # Normalize time to days for interpretable slope
                times_days = (times - times[0]) / DAY
                if times_days[-1] > 0:
                    # Simple linear regression: slope = cov(x,y) / var(x)
                    slope = np.polyfit(times_days, ratings, 1)[0]
                    rating_trend_slope = float(slope)
                else:
                    rating_trend_slope = 0.0
            else:
                rating_trend_slope = 0.0

            # Contest rank percentile avg
            # rank field in rating_history gives the user's rank; we don't have
            # total participants directly, so we use rank as a proxy (lower = better)
            if window_contests:
                ranks = [c.get("rank", 0) for c in window_contests]
                contest_rank_percentile_avg = float(np.mean(ranks))
            else:
                contest_rank_percentile_avg = 0.0

            # Solve speed avg (avg relativeTimeSeconds for AC contest submissions)
            contest_ac = [s for s in contest_subs if s["verdict"] == "OK"]
            if contest_ac:
                solve_times = [s.get("relativeTimeSeconds", 0) for s in contest_ac]
                solve_speed_avg = float(np.mean(solve_times)) / 60.0  # in minutes
            else:
                solve_speed_avg = 0.0

            # Difficulty ceiling (max problem rating solved)
            ac_ratings = [s["problem"].get("rating", 0) for s in ac_subs if s["problem"].get("rating")]
            difficulty_ceiling = max(ac_ratings) if ac_ratings else 0

            # Problem rating vs user rating (avg attempted / current, floor at 800 for unrated)
            prob_ratings = [s["problem"].get("rating") for s in window_subs if s["problem"].get("rating")]
            problem_rating_vs_user = (np.mean(prob_ratings) / max(800, user_rating)) if prob_ratings else 0.0

            # ═══════════════════════════════════════════════════
            # 4. TOPIC DIVERSITY FEATURES
            # ═══════════════════════════════════════════════════

            tag_ent = entropy(window_tags)
            unique_tags = len(set(window_tags))

            # New tags explored (tags in this window never seen before)
            window_tags_set = set(window_tags)
            new_tags_explored = len(window_tags_set - historical_tags_seen)

            # Tag concentration top 3
            tag_conc_top3 = top_k_concentration(window_tags)

            # ═══════════════════════════════════════════════════
            # 5. LEARNING EFFICIENCY FEATURES
            # ═══════════════════════════════════════════════════

            # Attempts per AC
            attempts = Counter(
                (s["problem"].get("name", ""), s["problem"].get("index", ""))
                for s in window_subs
            )
            attempts_per_ac = float(np.mean(list(attempts.values()))) if attempts else 0.0

            # First attempt AC rate
            # Group submissions by problem, check if first submission was AC
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

            # Avg debug time (for problems with WA->AC, time from first WA to final AC)
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
                    debug_times.append((last_ac - first_wa) / 60.0)  # in minutes

            avg_debug_time = float(np.mean(debug_times)) if debug_times else 0.0

            # ═══════════════════════════════════════════════════
            # 6. CONTEST BEHAVIOR FEATURES
            # ═══════════════════════════════════════════════════

            contest_count = len(window_contests)

            # Contest problems solved avg (AC problems per contest)
            if contest_subs:
                problems_per_contest = defaultdict(set)
                for s in contest_subs:
                    if s["verdict"] == "OK":
                        cid = s.get("contestId", s["problem"].get("contestId", ""))
                        pid = (cid, s["problem"].get("index", ""))
                        problems_per_contest[cid].add(pid)
                if problems_per_contest:
                    contest_problems_solved_avg = float(np.mean([len(v) for v in problems_per_contest.values()]))
                else:
                    contest_problems_solved_avg = 0.0
            else:
                contest_problems_solved_avg = 0.0

            # ═══════════════════════════════════════════════════
            # TARGET
            # ═══════════════════════════════════════════════════

            future_delta = label_contests[-1]["newRating"] - label_contests[0]["oldRating"]

            # ═══════════════════════════════════════════════════
            # ASSEMBLE ROW
            # ═══════════════════════════════════════════════════

            rows.append({
                "handle": handle,
                # Upsolving
                "upsolve_count": upsolve_count,
                "upsolve_ratio": upsolve_ratio,
                "upsolve_difficulty_delta": upsolve_difficulty_delta,
                "contest_to_practice_ratio": contest_to_practice_ratio,
                # Consistency
                "subs_90d": total_subs,
                "ac_ratio": ac_ratio,
                "active_days": active_days,
                "avg_gap_days": avg_gap_days,
                "max_gap_days": max_gap,
                "weekly_consistency": weekly_con,
                "streak_max": streak,
                "speed_vs_growth_ratio": speed_vs_growth,
                # IQ
                "base_rating": base_rating,
                "rating_delta_90d": rating_delta,
                "rating_volatility": rating_volatility,
                "rating_trend_slope": rating_trend_slope,
                "contest_rank_percentile_avg": contest_rank_percentile_avg,
                "solve_speed_avg": solve_speed_avg,
                "difficulty_ceiling": difficulty_ceiling,
                "problem_rating_vs_user_rating": problem_rating_vs_user,
                # Topic Diversity
                "tag_entropy": tag_ent,
                "unique_tags_count": unique_tags,
                "new_tags_explored": new_tags_explored,
                "tag_concentration_top3": tag_conc_top3,
                # Learning Efficiency
                "attempts_per_ac": attempts_per_ac,
                "first_attempt_ac_rate": first_attempt_ac_rate,
                "avg_debug_time": avg_debug_time,
                # Contest Behavior
                "contest_count": contest_count,
                "contest_problems_solved_avg": contest_problems_solved_avg,
                # Target
                "future_rating_delta_30d": future_delta,
            })

            t += STEP_DAYS * DAY

print("Saving CSV...")

pd.DataFrame(rows).to_csv("cf_ml_dataset.csv", index=False)

print(f"Done. {len(rows)} samples written.")
