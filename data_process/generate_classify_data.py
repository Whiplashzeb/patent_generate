import os
import json
import re
import random

json_dir = "../../patent_data/json/"
classify_dir = "../../patent_data/classify/"

generate_data = list()


def get_range(r):
    ran = re.split(r"-+|~|至", r)
    # if r.find("至") != -1:
    #     ran = r.split("至")
    start, end = int(ran[0]), int(ran[1])
    result = list()
    for i in range(start, end + 1):
        result.append(str(i))
    return result


def get_dependent(reference):
    match = re.search(r"权利要求\d+所述", reference)
    if match is not None:
        return [match.group()[4:-2]]
    match = re.search(r"权利要求\d+[或,、]\d+所述", reference)
    if match is not None:
        return re.split(r"[或,、]", match.group()[4:-2])
        # return [match.group()[4:-2].split("或")]
    match = re.search(r"权利要求\d+(-+|至|~)\d+", reference)
    if match is not None:
        range = match.group()[4:]
        return get_range(range)
    match = re.search(r"权利要求\d+、\d+或\d+", reference)
    if match is not None:
        return re.split(r"[或,、]", match.group()[4:])


def load_json(file_list):
    global generate_data
    for file_name in file_list:
        independ_dict = dict()
        depend_dict = dict()
        positive = set()

        json_file = os.path.join(json_dir, file_name)
        with open(json_file) as fp:
            patent_json = json.load(fp)

            independents = patent_json["independent"]
            # 判断文件是否符合规范, 开头以数字进行编号
            if len(independents):
                first = independents[0]
                if len(first[0]):
                    if not first[0][0].isdigit():
                        continue
                else:
                    if not first[1][0].isdigit():
                        continue
            else:
                continue
            dependents = patent_json["dependent"]

            for independ in independents:
                preamble, character = independ[0], independ[1]
                if character.startswith("，") or character.startswith("：") or character.startswith(":"):
                    character = character[1:]
                if len(preamble) and len(character) <= 128:
                    if preamble[0].isdigit():
                        independ_dict[preamble[0]] = character.replace("\n", "")

            for depend in dependents:
                reference, limited = depend[0], depend[1]
                if limited.startswith("，") or limited.startswith("：") or limited.startswith(":"):
                    limited = limited[1:]
                if len(reference) and len(limited) <= 128:
                    if reference[0].isdigit():
                        depend_dict[reference[0]] = limited.replace("\n", "")
                        ran = get_dependent(reference)
                        if ran is not None:
                            for i in ran:
                                positive.add((i, reference[0]))

            for in_k, in_v in independ_dict.items():
                for de_k, de_v in depend_dict.items():
                    if (in_k, de_k) in positive:
                        generate_data.append((1, in_v, de_v))
                    else:
                        generate_data.append((0, in_v, de_v))

            for f_k, f_v in depend_dict.items():
                for s_k, s_v in depend_dict.items():
                    if f_k == s_k:
                        continue
                    if (f_k, s_k) in positive:
                        generate_data.append((1, f_v, s_v))
                    elif (s_k, f_k) in positive:
                        generate_data.append((-1, f_v, s_v))
                    else:
                        generate_data.append((0, f_v, s_v))


def generate_file(dev_per, test_per):
    global generate_data
    l = len(generate_data)
    train_per = 1 - dev_per - test_per
    random.shuffle(generate_data)
    print(l)

    with open(os.path.join(classify_dir, "train.tsv"), "w") as fp:
        fp.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for data in generate_data[:int(l * train_per)]:
            fp.write("%d\t1\t1\t%s\t%s\n" % (data[0], data[1], data[2]))
    with open(os.path.join(classify_dir, "dev.tsv"), "w") as fp:
        fp.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for data in generate_data[int(l * train_per): int(l * (train_per + dev_per))]:
            fp.write("%d\t1\t1\t%s\t%s\n" % (data[0], data[1], data[2]))
    with open(os.path.join(classify_dir, "test.tsv"), "w") as fp:
        fp.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for data in generate_data[int(l * (train_per + dev_per)):]:
            fp.write("%d\t1\t1\t%s\t%s\n" % (data[0], data[1], data[2]))


if __name__ == "__main__":
    file_list = os.listdir(json_dir)
    load_json(file_list)
    generate_file(0.03, 0.02)
