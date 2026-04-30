# Competitive Programming Growth Reliability Modeling

This project builds a behavioral machine learning dataset from public Codeforces data to estimate the *sustainability and reliability* of a user's improvement strategy.

Rather than directly framing the task as rating prediction, the model learns a **Growth Reliability Score** — a proxy for how stable and effective a user's learning patterns are — derived from recent contest participation, practice behavior, and topic progression.

Under the hood, the target variable is the user's rating change over the following 30 days.

<h2 align="center">Project Poster</h2>
<p align="center">
  <img src="./poster.png" alt="A3 Project Poster" width="600"/>
</p>

---

## Overview

Competitive programming improvement is not purely a function of raw activity.  
It depends on:

- Upsolving behavior
- Consistency & practice regularity
- Cognitive ability / problem-solving speed
- Topic diversity & exploration
- Post-contest learning behavior

This project attempts to quantify these behaviors using temporal feature engineering and supervised learning.

Each sample represents:

> User behavior over 90 days → Rating change over the next 30 days

Sliding windows are used so each user contributes multiple samples.

---

## Dataset Files

### `codeforces_dataset.jsonl`

Raw scraped data (not included in this repository).

Each line corresponds to one Codeforces user and contains:

- `handle`
- `profile`
- `rating_history`
- `submissions`

This file preserves the full event history.

⚠️ **Note:** This file is excluded via `.gitignore` due to size and can be regenerated using the provided scraping scripts.

---

### `cf_ml_dataset.csv`

Processed machine-learning dataset.

Each row represents a single temporal window:

> 90-day behavioral features → next 30-day rating delta

This is the main dataset used for modeling and is included in the repository.

---

## Feature Description (`cf_ml_dataset.csv`)

### Upsolving (4 features)

| Column | Description |
|--------|-------------|
| `upsolve_count` | Number of accepted practice submissions on contest problems |
| `upsolve_ratio` | Fraction of contest problems later revisited via practice |
| `upsolve_difficulty_delta` | Avg rating of upsolved problems minus user's current rating |
| `contest_to_practice_ratio` | Ratio of practice submissions to contest submissions |

### Consistency (8 features)

| Column | Description |
|--------|-------------|
| `subs_90d` | Total submissions in feature window |
| `ac_ratio` | Fraction of accepted submissions |
| `active_days` | Number of distinct days with ≥1 submission |
| `avg_gap_days` | Average gap (in days) between active days |
| `max_gap_days` | Longest inactivity gap (days) |
| `weekly_consistency` | Std deviation of weekly submission counts (lower = more consistent) |
| `streak_max` | Longest consecutive-day submission streak |
| `speed_vs_growth_ratio` | Ratio of submissions on familiar tags to new tags |

### IQ / Cognitive Ability (8 features)

| Column | Description |
|--------|-------------|
| `base_rating` | Rating at the start of the feature window |
| `rating_delta_90d` | Rating change during the feature window |
| `rating_volatility` | Std deviation of per-contest rating changes |
| `rating_trend_slope` | Linear regression slope of rating over time (rating/day) |
| `contest_rank_percentile_avg` | Average rank across contests in the window |
| `solve_speed_avg` | Average time (minutes) to solve contest problems |
| `difficulty_ceiling` | Maximum problem rating successfully solved |
| `problem_rating_vs_user_rating` | Avg attempted problem rating / user rating |

### Topic Diversity (4 features)

| Column | Description |
|--------|-------------|
| `tag_entropy` | Shannon entropy over problem tags |
| `unique_tags_count` | Number of distinct tags encountered |
| `new_tags_explored` | Tags attempted for the first time in this window |
| `tag_concentration_top3` | Fraction of submissions in top-3 most common tags |

### Learning Efficiency (3 features)

| Column | Description |
|--------|-------------|
| `attempts_per_ac` | Average attempts per unique problem |
| `first_attempt_ac_rate` | Fraction of problems solved on the first try |
| `avg_debug_time` | Avg time (minutes) from first WA to final AC per problem |

### Contest Behavior (2 features)

| Column | Description |
|--------|-------------|
| `contest_count` | Number of rated contests participated in |
| `contest_problems_solved_avg` | Average problems solved per contest |

### Target

| Column | Description |
|--------|-------------|
| `future_rating_delta_30d` | Rating change in the next 30 days |

---

## Dataset Construction

- Feature window: 90 days  
- Label window: next 30 days  
- Sliding step: 30 days  

Raw submission and contest histories are streamed line-by-line to avoid memory overload.

All features are aggregated from event-level data into fixed-length numeric vectors suitable for classical machine learning models.

---

## Intended Use

The dataset is designed for:

- Regression (predicting future rating delta)
- Classification (growth / stagnation / decline)
- Clustering (identifying successful learning patterns)
- Behavioral modeling of competitive programming strategies

The output may be interpreted as a **Growth Reliability Score** reflecting how sustainable a user's current approach is.

---

## Notes

- All data comes from the public Codeforces API.
- No private information is included.
- Rating trajectories are inherently noisy due to real-world factors (burnout, inactivity, etc.).
- Raw JSONL data can be regenerated using the scraping scripts.

---

## License

Educational and research use only.
