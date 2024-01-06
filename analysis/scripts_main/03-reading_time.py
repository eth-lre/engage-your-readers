import json
import sys
sys.path.append("analysis")
import loader
import collections
import numpy as np
import matplotlib.pyplot as plt

data = loader.load_by_user_prolific(min_phases=5)

def uid_to_group(uid):
    for group in ["control", "authentic", "generated"]:
        if uid.endswith("_" + group):
            return group

    raise Exception("Unknown UID")

paragraph_word_counts = {}
for uid, userdata in data.items():
    # the uid in the data also contains the prolific pid part
    uid = uid.split("___")[1]
    paragraph_word_counts[uid] = 0

for uid in paragraph_word_counts:
    article = json.load(open(f"annotation_ui/web/queues/{uid}.jsonl", "r"))["article"]
    paragraph_word_counts[uid] = [
        len(paragraph.strip().split())
        for paragraph in article.split("<p>")
    ]

data_agg = collections.defaultdict(lambda: collections.defaultdict(list))
for uid, userdata in data.items():
    group = uid_to_group(uid)

    # the uid in the data also contains the prolific pid part
    uid = uid.split("___")[1]
    reading_times = [
        value
        for key, value in userdata[2].items()
        if key.startswith("finish_reading_")
    ]
    reading_times.sort()
    reading_times = [
        y-x for x, y in zip(reading_times, reading_times[1:])
    ]

    for paragraph_i, (word_count, duration) in enumerate(zip(paragraph_word_counts[uid], reading_times)):
        data_agg[group][paragraph_i].append(duration/word_count)

fig = plt.figure(figsize=(3, 2))
ax = plt.gca()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

GROUP_COLORS = {
    "authentic": "#499e67",
    "control": "#b6b6b6",
    "generated": "#9e4980",
}

for group, group_data in data_agg.items():
    print(f"- {group:>10}: ", end="")
    y_vals = []
    for paragraph_i, paragraph_v in group_data.items():
        y_vals.append(np.average(paragraph_v) / 1000)
        print(f"{np.average(paragraph_v)/1000:.3f}s ", end="")
    print()
    plt.plot(
        y_vals[1:],
        label=f"{group.capitalize()} ({np.average(y_vals):.3f}s)",
        color=GROUP_COLORS[group],
    )

plt.legend(fancybox=False)
plt.xlabel("Paragraph position")
plt.ylabel("Paragraph\ntime (s)", labelpad=-20)
plt.tight_layout(pad=0.2)
plt.savefig("computed/reading_time.pdf")
plt.show()
