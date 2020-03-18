import os
import csv

classify_dir = "../../../patent_data/augment_data/"


def load_file(file_name):
    statistics = {
        "num": 0,
        "1": 0,
        "0": 0,
        "-1": 0,
        "a_len": [0 for _ in range(9)],
        "b_len": [0 for _ in range(9)]
    }

    with open(os.path.join(classify_dir, file_name), "r", encoding="utf-8-sig") as fp:
        all = list(csv.reader(fp, delimiter="\t"))
        statistics["num"] = len(all)
        for line in all[1:]:
            label, text_a, text_b = line[0], line[3], line[4]
            statistics[label] += 1
            statistics["a_len"][len(text_a) // 16] += 1
            statistics["b_len"][len(text_b) // 16] += 1

    return statistics

if __name__ == "__main__":
    statistics = load_file("train.tsv")

    for key in statistics.keys():
        print("%s is %s" % (key, str(statistics[key])))