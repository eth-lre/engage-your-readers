def load_by_user(path="data.jsonl", allowlist=None, denylist=None, min_phases=None):
    import json
    import collections

    raw_text = open(path, "r").read()
    # hack handling of data save race
    raw_text = raw_text.replace("}{", "}\n{").rstrip()
    data = [json.loads(x.rstrip()) for x in raw_text.split("\n")]

    # apply filters
    if allowlist is not None:
        data = [
            x for x in data
            if any(needle in x["uid"] for needle in allowlist)
        ]
    if denylist is not None:
        data = [
            x for x in data
            if not any(needle in x["uid"] for needle in denylist)
        ]

    data_clean = collections.defaultdict(dict)
    for line in data:
        data_clean[line["prolific_pid"] + "___" + line["uid"]][line["phase"]] = line

    if min_phases:
        data_clean = {
            uid: vals
            for uid, vals in data_clean.items()
            if len(vals) >= min_phases
        }

    return data_clean

def load_by_user_prolific(path="data.jsonl", min_phases=None):
    import json
    import collections

    raw_text = open(path, "r").read()
    # hack handling of data save race
    raw_text = raw_text.replace("}{", "}\n{").rstrip()
    data = [json.loads(x.rstrip()) for x in raw_text.split("\n")]

    data_clean = collections.defaultdict(dict)
    for line in data:
        if "prolific_pid" not in line:
            continue

        data_clean[line["prolific_pid"] + "___" + line["uid"]][line["phase"]] = line

    if min_phases:
        data_clean = {
            uid: vals
            for uid, vals in data_clean.items()
            if len(vals) >= min_phases
        }

    return data_clean