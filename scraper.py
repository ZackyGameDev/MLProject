import requests
import json
import random
import time
import os
from tqdm import tqdm

BASE = "https://codeforces.com/api"
SLEEP = 1.5

USERS_PER_BUCKET = 500
OUTFILE = "codeforces_dataset.jsonl"
RATED_USERS_CACHE = "rated_users.json"


def cf(endpoint, params=None):
    r = requests.get(BASE + endpoint, params=params)
    time.sleep(SLEEP)
    r.raise_for_status()
    data = r.json()
    if data["status"] != "OK":
        raise Exception(data)
    return data["result"]


# Load already saved handles (resume support)
done = set()
if os.path.exists(OUTFILE):
    with open(OUTFILE, "r", encoding="utf8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["handle"])
            except:
                pass

print("Already downloaded:", len(done))


# Rated users cache
if os.path.exists(RATED_USERS_CACHE):
    print("Loading rated users from cache...")
    with open(RATED_USERS_CACHE, "r", encoding="utf8") as f:
        users = json.load(f)
else:
    print("Fetching rated user list from Codeforces...")
    users = cf("/user.ratedList", {"activeOnly": "true"})
    with open(RATED_USERS_CACHE, "w", encoding="utf8") as f:
        json.dump(users, f)
    print("Saved rated users to cache.")

# Bucketing 
buckets = {
    "newbie": [],
    "pupil": [],
    "specialist": [],
    "expert": [],
    "master+": []
}

for u in users:
    r = u.get("rating", 0)
    if r < 1000:
        buckets["newbie"].append(u["handle"])
    elif r < 1400:
        buckets["pupil"].append(u["handle"])
    elif r < 1800:
        buckets["specialist"].append(u["handle"])
    elif r < 2200:
        buckets["expert"].append(u["handle"])
    else:
        buckets["master+"].append(u["handle"])

print("\nBucket sizes:")
for k in buckets:
    print(k, len(buckets[k]))

# Sampling

selected = []

for k in buckets:
    selected += random.sample(
        buckets[k],
        min(USERS_PER_BUCKET, len(buckets[k]))
    )

# Remove already downloaded
selected = [h for h in selected if h not in done]

print("\nRemaining users:", len(selected))

with open(OUTFILE, "a", encoding="utf8") as f:
    for handle in tqdm(selected):
        try:
            info = cf("/user.info", {"handles": handle})[0]
            rating = cf("/user.rating", {"handle": handle})
            subs = cf("/user.status", {"handle": handle, "count": 10000})

            row = {
                "handle": handle,
                "profile": info,
                "rating_history": rating,
                "submissions": subs
            }

            f.write(json.dumps(row) + "\n")
            f.flush()

        except Exception as e:
            print("Failed:", handle, e)

print("\nDone.")
