import os
import json

summary_dir = "../../patent_data/summary/"


def load_json(src_file, tgt_file):
    statistics = {
        "num": 0,
        "src_len": [0 for _ in range(10)],
        "tgt_len": [0 for _ in range(10)],
        "longer": 0
    }
    with open(os.path.join(summary_dir, src_file)) as src, open(os.path.join(summary_dir, tgt_file)) as tgt:
        for s_line, t_line in zip(src.readlines(), tgt.readlines()):
            s_line = s_line.replace(" ", "").strip()
            t_line = t_line.replace(" ", "").strip()
            len_s = len(s_line)
            len_t = len(t_line)

            statistics["num"] += 1
            statistics["src_len"][len_s // 128] += 1
            statistics["tgt_len"][len_t // 64] += 1
            if len_t > len_s:
                statistics["longer"] += 1

    return statistics

if __name__ == "__main__":
    statistics = load_json("test.src", "test.tgt")
    for key in statistics.keys():
        print("%s is %s" % (key, str(statistics[key])))
