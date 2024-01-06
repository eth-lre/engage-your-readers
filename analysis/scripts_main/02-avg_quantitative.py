import sys
sys.path.append("analysis")
import loader
import collections
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("computed/microbars/", exist_ok=True)

data = loader.load_by_user_prolific(min_phases=5)


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
    "engaging": factory_extract_likert(4, "likert_engaging#radio"),
    "understanding": factory_extract_likert(4, "likert_understanding#radio"),
    "overall": factory_extract_likert(4, "likert_overall#radio"),
    "question_relevant": factory_get_question_likert(5),
    "question_distracting": factory_get_question_likert(6),
    "question_important": factory_get_question_likert(7),
}


def uid_to_group(uid):
    for group in ["control", "authentic", "generated"]:
        if uid.endswith("_" + group):
            return group

    raise Exception("Unknown UID")


def visualize_distribution(data, filename):
    plt.figure(figsize=(0.5, 0.2))
    ax = plt.gca()

    ax.spines[['right', 'top', 'left']].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    plt.hist(data, color="black", bins=[1, 2, 3, 4, 5, 6])
    plt.ylim(0, len(data))
    plt.xlim(1, 6)
    plt.tight_layout(pad=0.1)
    plt.savefig("computed/microbars/" + filename + ".pdf")
    

for extractor_name, extractor_func in EXTRACTORS.items():
    print(extractor_name)
    data_agg = collections.defaultdict(list)
    for userdata in data.values():
        group = uid_to_group(userdata[0]["uid"])
        if group == "control":
            continue
        data_agg[group].append(extractor_func(userdata))

    for group, group_data in data_agg.items():
        filename = extractor_name + "___" + group
        visualize_distribution(
            group_data,
            filename=filename
        )
        print(f"- {group:>10} {np.average(group_data):.1f} \\includegraphics[height=4mm]{{figures/microbars/{filename}.pdf}}")

    print()
