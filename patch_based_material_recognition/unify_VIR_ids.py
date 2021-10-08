import json

with open("/local/temporary/DATASET/info/id_to_OWN.json", "r") as f:
    lll = dict(json.load(f))
    groups = list()
    cat_groups = list()
    visited = set()
    values = list(lll.values())
    keys = list(lll.keys())
    last_group_index = -1
    for i in range(len(values)):
        temp_val = values[i]
        # print(len(temp_val.split(" - ")))
        cat, mat = temp_val.split(" - ")
        # print(temp_val.split(" - "))
        for j in range(i, len(values)):
            temp_val2 = values[j]
            cat2, mat2 = temp_val2.split(" - ")
            if cat2 == cat:
                if cat not in visited:
                    groups.append({i, j})
                    last_group_index += 1
                    visited.add(cat)
                    cat_groups.append(cat)
                else:
                    if j not in groups[last_group_index]:
                        groups[last_group_index].add(j)
    ret_id_to_cat_id = dict()
    ret_id_to_cat = dict()
    for k, vs in enumerate(groups):
        # ret_id_to_cat_id[k] = vs  # index : {original id, original id, ...}
        for v in vs:
            ret_id_to_cat_id[v] = k
            ret_id_to_cat[v] = cat_groups[k]
    # print(ret_id_to_cat)
    # print()
    print()
    with open('OWN_to_categorized_id.json', 'w') as f:
        print('OWN_to_categorized_id.json <=', ret_id_to_cat_id)
        print()
        json.dump(ret_id_to_cat_id, f)
    with open('OWN_to_categorized_str.json', 'w') as f:
        print('OWN_to_categorized_str.json <=', ret_id_to_cat)
        print()
        json.dump(ret_id_to_cat, f)

    # print(groups)
    # for i in range(len(values)):
    #     temp_val = values[i]
    #     for ci in range(len(temp_val)):
    #         for j in range(len(values)):
    #             temp_val2 = values[j]
    #             if temp_val2[0: ci+1] == temp_val[0: ci+1]:
    # print(lll.values())
    # print(lll.keys())


def str_intersection(s1, s2):
    out = ""
    for c in s1:
        if c in s2 and c not in out:
            out += c
    return out


