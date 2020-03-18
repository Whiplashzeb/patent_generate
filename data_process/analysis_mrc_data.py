import os
import json

mrc_dir = "../../patent_data/mrc/mrc_add_title/"


def load_json(file_name):
    statistics = {
        "passages": 0,
        "paragraphs": 0,
        "pair": 0,
        "len": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    mrc_file = os.path.join(mrc_dir, file_name)
    with open(mrc_file) as fp:
        mrc_json = json.load(fp)

        passages = mrc_json["data"]
        statistics["passages"] = len(passages)
        for passage in passages:
            paragraphs = passage["paragraphs"]
            statistics["paragraphs"] += len(paragraphs)
            for concrete in paragraphs:
                qas = concrete["qas"]
                statistics["pair"] += len(qas)
                for qa in qas:
                    question = qa["question"]
                    answer = qa["answers"][0]["text"]

                    l = len(answer)
                    if l > 128:
                        print(answer)
                    statistics["len"][l // 16] += 1
    return statistics


if __name__ == "__main__":
    file_name = "test_v1.1.json"
    statistics = load_json(file_name)

    for key in statistics.keys():
        print("%s is %s" % (key, str(statistics[key])))
