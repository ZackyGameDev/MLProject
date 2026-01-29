# Codeforces Rating Growth Prediction Dataset

This repository contains a machine-learning ready dataset constructed from public Codeforces user data.  
The goal of this project is to model and predict short-term rating changes based on recent user behavior.

Each row in the dataset represents a single user over a fixed time window, with aggregated behavioral features extracted from submissions and contest participation.

---

## Dataset Files

### `codeforces_dataset.jsonl`

Raw scraped data (not included in this repository).

Each line corresponds to one Codeforces user and contains:

- `handle`: Username
- `profile`: Static profile information (rating, maxRating, registration time, etc.)
- `rating_history`: List of rated contest participations
- `submissions`: List of all submissions (practice + contest)

⚠️ **Note:** This raw JSONL file is excluded via `.gitignore` due to its size and because it can be regenerated using the provided scraping scripts.

---

### `cf_ml_dataset.csv`

Processed machine-learning dataset.

Each row represents:

> User behavior over 90 days → Rating change over the following 30 days

This is the main dataset used for modeling and is included in the repository.

---

## Feature Description (`cf_ml_dataset.csv`)

| Column | Description |
|--------|-------------|
| `handle` | Codeforces username (identifier only, not used for training) |
| `subs_90d` | Total number of submissions in the 90-day feature window |
| `ac_ratio` | Fraction of submissions that were Accepted (OK) |
| `avg_problem_rating` | Average difficulty rating of attempted problems |
| `tag_entropy` | Entropy of problem tags, measuring topic diversity |
| `attempts_per_ac` | Average number of attempts per problem |
| `contest_count` | Number of rated contests participated in |
| `rating_delta_90d` | Rating change during the 90-day feature window |
| `max_gap_days` | Largest gap (in days) between active submission days |
| `future_rating_delta_30d` | Rating change in the next 30 days (target variable) |

---

## Feature Categories

### Activity
- `subs_90d`
- `contest_count`
- `max_gap_days`

Measure how frequently and consistently a user practices.

---

### Skill / Difficulty
- `ac_ratio`
- `avg_problem_rating`

Estimate problem-solving efficiency and difficulty level.

---

### Learning Pattern
- `tag_entropy`
- `attempts_per_ac`

Capture topic diversity and learning efficiency.

---

### Momentum
- `rating_delta_90d`

Represents recent rating trajectory.

---

### Target

- `future_rating_delta_30d`

This is the supervised learning target: future rating change.

---

## Dataset Construction

The dataset is generated using sliding temporal windows:

- Feature window: 90 days  
- Label window: next 30 days  
- Step size: 30 days  

This allows each user to contribute multiple samples, preserving temporal dynamics and increasing dataset size.

Raw Codeforces API data is aggregated into fixed-length numerical feature vectors suitable for supervised machine learning.

---

## Intended Use

The dataset is designed for:

- Regression (predicting future rating delta)
- Classification (growth / stagnation / decline)
- Feature importance analysis
- Behavioral modeling of competitive programming progress

---

## Notes

- All data is collected from the public Codeforces API.
- No private information is included.
- Ratings are inherently noisy due to real-world factors such as inactivity or burnout.
- The raw JSONL file can be regenerated using the scraping scripts in this repository.

---

## License

This dataset is for educational and research purposes only.
