# Competitive Programming Growth Reliability Modeling

This project builds a behavioral machine learning dataset from public Codeforces data to estimate the *sustainability and reliability* of a user’s improvement strategy.

Rather than directly framing the task as rating prediction, the model learns a **Growth Reliability Score** — a proxy for how stable and effective a user’s learning patterns are — derived from recent contest participation, practice behavior, and topic progression.

Under the hood, the target variable is the user’s rating change over the following 30 days.

---

## Overview

Competitive programming improvement is not purely a function of raw activity.  
It depends on:

- Consistency
- Recovery from failure
- Difficulty progression
- Topic mastery
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

| Column | Description |
|--------|-------------|
| `handle` | User identifier (not used for training) |
| `subs_90d` | Total submissions in feature window |
| `ac_ratio` | Fraction of accepted submissions |
| `avg_problem_rating` | Average difficulty of attempted problems |
| `tag_entropy` | Topic diversity (entropy over problem tags) |
| `attempts_per_ac` | Average attempts per problem |
| `contest_count` | Rated contests participated in |
| `rating_delta_90d` | Rating change during feature window |
| `max_gap_days` | Longest inactivity gap |
| `higher_difficulty_ac` | Count of accepted problems harder than current rating |
| `avg_recovery_minutes` | Average time to resubmit after failure |
| `submission_density` | Submissions per active day |
| `rating_gap_variance` | Variance in difficulty of attempted problems |
| `avg_tag_improvement` | Average improvement in tag-specific success rate |
| `future_rating_delta_30d` | Rating change in next 30 days (target) |

---

## Feature Categories

### Activity & Consistency
- `subs_90d`
- `contest_count`
- `max_gap_days`
- `submission_density`

Measure engagement and regularity.

---

### Skill & Difficulty Progression
- `avg_problem_rating`
- `higher_difficulty_ac`
- `rating_gap_variance`

Capture difficulty stretching and comfort-zone behavior.

---

### Learning Efficiency
- `ac_ratio`
- `attempts_per_ac`
- `avg_recovery_minutes`

Model how users respond to failure.

---

### Topic Mastery
- `tag_entropy`
- `avg_tag_improvement`

Measure breadth and improvement across problem domains.

---

### Momentum
- `rating_delta_90d`

Represents recent rating trajectory.

---

### Target

- `future_rating_delta_30d`

Supervised learning target representing short-term rating change.

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

The output may be interpreted as a **Growth Reliability Score** reflecting how sustainable a user’s current approach is.

---

## Notes

- All data comes from the public Codeforces API.
- No private information is included.
- Rating trajectories are inherently noisy due to real-world factors (burnout, inactivity, etc.).
- Raw JSONL data can be regenerated using the scraping scripts.

---

## License

Educational and research use only.
