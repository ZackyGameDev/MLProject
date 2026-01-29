import requests
import json
import random
import time
import os
from tqdm import tqdm

BASE = "https://codeforces.com/api"
SLEEP = 1.2

USERS_PER_BUCKET = 50
OUTFILE = "codeforces_dataset.jsonl"


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

print("Fetching rated user list...")
users = cf("/user.ratedList", {"activeOnly": "true"})

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

selected = []

for k in buckets:
    selected += random.sample(
        buckets[k],
        min(USERS_PER_BUCKET, len(buckets[k]))
    )

# Remove already downloaded
selected = [h for h in selected if h not in done]

print("Remaining users:", len(selected))

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

print("Done.")
