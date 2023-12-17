import sys
sys.path.append("analysis")
import loader
import collections
import numpy as np

data = loader.load_by_user(allowlist=["quant", "pogos", "jacon"], denylist=["uxx"])

def factory_extract_likert(phase, key):
    def func(userdata):
        return float(userdata[phase][key])
    return func

EXTRACTORS = {
    "satisfaction": factory_extract_likert(4, "likert_satisfaction#radio"),
    "content": factory_extract_likert(4, "likert_content#radio"),
    "relevance": factory_extract_likert(4, "likert_relevance#radio"),
    "frequency": factory_extract_likert(4, "likert_frequency#radio"),
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

for extractor_name, extractor_func in EXTRACTORS.items():
    print(extractor_name)
    data_agg = collections.defaultdict(list)
    for userdata in data.values():
        group = uid_to_group(userdata[0]["uid"])
        if group == "control":
            continue
        data_agg[group].append(extractor_func(userdata))

    for group, group_data in data_agg.items():
        print(f"- {group:>10} {np.average(group_data):.1f}")

    print()