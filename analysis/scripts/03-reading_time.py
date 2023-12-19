import sys
sys.path.append("analysis")
import loader
import collections
import numpy as np
import matplotlib.pyplot as plt

data = loader.load_by_user(allowlist=["quant", "pogos", "jacon"], denylist=["uxx"])

def factory_extract_likert(phase, key):
    def func(userdata):
        return float(userdata[phase][key])
    return func

def factory_get_question_likert(phase):
    def func(userdata):
        return np.average([
            float(userdata[phase][key])
            for key in userdata[phase] if key.endswith("#radio")
        ])
    return func

EXTRACTORS = {
    "satisfaction": factory_extract_likert(4, "likert_satisfaction#radio"),
    "content": factory_extract_likert(4, "likert_content#radio"),
    "relevance": factory_extract_likert(4, "likert_relevance#radio"),
    "frequency": factory_extract_likert(4, "likert_frequency#radio"),
    "question_helpful": factory_get_question_likert(5),
    "question_distracting": factory_get_question_likert(6),
    "question_relevant": factory_get_question_likert(7),
}

def uid_to_group(uid):
    if "quant" in uid:
        return "control"
    elif "pogos" in uid:
        return "authentic"
    elif "jacon" in uid:
        return "generated"
    else:
        raise Exception("Unknown UID")


data_agg = collections.defaultdict(lambda: collections.defaultdict(list))
for userdata in data.values():
    group = uid_to_group(userdata[0]["uid"])

    prev_time = float(userdata[2]["phase_start"])
    for key, value in userdata[2].items():
        if not key.startswith("finish_reading_"):
            continue
        paragraph_i = int(key.removeprefix("finish_reading_"))
    
        data_agg[group][paragraph_i].append(value-prev_time)
        prev_time = value

fig = plt.figure(figsize=(3,2))
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
        y_vals.append(np.average(paragraph_v)/1000)
        print(f"{np.average(paragraph_v)/1000:.0f}s ", end="")
    print()
    plt.plot(
        y_vals,
        label=f"{group.capitalize()} ({np.average(y_vals):.0f}s)",
        color=GROUP_COLORS[group],
    )

plt.legend(fancybox=False)
plt.xlabel("Paragraph position")
plt.ylabel("Paragraph\ntime (s)", labelpad=-20)
plt.xticks(
    [0, 2, 4, 6, 8, 10],
    [1, 3, 5, 7, 9, 11],
)
plt.yticks([25, 50, 150])
plt.tight_layout(pad=0.2)
plt.savefig("computed/reading_time.pdf")
plt.show()