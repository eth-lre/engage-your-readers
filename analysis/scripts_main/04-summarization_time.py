import json
import sys
sys.path.append("analysis")
import loader
import collections
import numpy as np

data = loader.load_by_user_prolific(min_phases=5)

def uid_to_group(uid):
    for group in ["control", "authentic", "generated"]:
        if uid.endswith("_" + group):
            return group

    raise Exception("Unknown UID")

data_times = collections.defaultdict(list)
for uid, userdata in data.items():
    group = uid_to_group(uid)

    data_times[group].append(userdata[4]["phase_start"]-userdata[3]["phase_start"])

data_words = collections.defaultdict(list)
for uid, userdata in data.items():
    group = uid_to_group(uid)
    data_words[group].append(len(userdata[3]["text_0"].strip().split()))

GROUP_COLORS = {
    "authentic": "#499e67",
    "control": "#b6b6b6",
    "generated": "#9e4980",
}

for group, group_data in data_times.items():
    print(f"- {group:>10}: ", end="")
    print(f"{np.average(group_data)/1000:.0f}s ", end="")
    print()


print()
for group, group_data in data_words.items():
    print(f"- {group:>10}: ", end="")
    print(f"{np.average(group_data):.0f}w ", end="")
    print()
