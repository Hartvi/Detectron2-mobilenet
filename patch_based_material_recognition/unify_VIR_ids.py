import json

with open("/local/temporary/DATASET/info/id_to_OWN.json", "r") as f:
    lll = json.load(f)

print(lll)


def str_intersection(s1, s2):
    out = ""
    for c in s1:
        if c in s2 and c not in out:
            out += c
    return out


