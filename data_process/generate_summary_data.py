import os
import json
import random

json_dir = "../../patent_data/json/"
summary_dir = "../../patent_data/summary/"

generate_data = list()


def load_json(file_list):
    global generate_data
    for file_name in file_list:
        json_file = os.path.join(json_dir, file_name)
        with open(json_file) as fp:
            patent = json.load(fp)

            invention_content = patent["invention_content"]
            independents = patent["independent"]
            dependents = patent["dependent"]

            for independ in independents:
                preamble, character, (start, end) = independ[0], independ[1], (int(independ[2][0]), int(independ[2][0]))
                if preamble != "" and len(character) <= 512 and start != end and end - start <= 1024:
                    src = invention_content[start:end].replace("\n", "")
                    tgt = character.replace("\n", "")
                    if tgt.startswith("，") or tgt.startswith("：") or tgt.startswith(":"):
                        tgt = tgt[1:]
                    generate_data.append([src, tgt])

            for depend in dependents:
                reference, limited, (start, end) = depend[0], depend[1], (int(depend[2][0]), int(depend[2][1]))
                if reference != "" and len(limited) <= 512 and start != end and end - start <= 1024:
                    src = invention_content[start:end].replace("\n", "")
                    tgt = limited.replace("\n", "")
                    if tgt.startswith("，") or tgt.startswith("：") or tgt.startswith(":"):
                        tgt = tgt[1:]
                    generate_data.append([src, tgt])


def generate_file(dev_per, test_per):
    global generate_data
    l = len(generate_data)
    random.shuffle(generate_data)
    print(l)
    train_per = 1 - dev_per - test_per
    with open(os.path.join(summary_dir, "train.src"), "w") as t_src, open(os.path.join(summary_dir, "train.tgt"), "w") as t_tgt:
        for src, tgt in generate_data[:int(l * train_per)]:
            t_src.write(" ".join(src) + '\n')
            t_tgt.write(" ".join(tgt) + '\n')
    with open(os.path.join(summary_dir, "dev.src"), "w") as d_src, open(os.path.join(summary_dir, "dev.tgt"), "w") as d_tgt:
        for src, tgt in generate_data[int(l * train_per):int(l * (train_per + dev_per))]:
            d_src.write(" ".join(src) + '\n')
            d_tgt.write(" ".join(tgt) + '\n')
    with open(os.path.join(summary_dir, "test.src"), "w") as t_src, open(os.path.join(summary_dir, "test.tgt"), "w") as t_tgt:
        for src, tgt in generate_data[int(l * (train_per + dev_per)):]:
            t_src.write(" ".join(src) + '\n')
            t_tgt.write(" ".join(tgt) + '\n')


if __name__ == "__main__":
    file_list = os.listdir(json_dir)
    load_json(file_list)
    generate_file(0.03, 0.02)
