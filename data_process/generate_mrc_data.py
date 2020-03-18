import os
import json
import random

json_dir = "../../patent_data/json/"
mrc_dir = "../../patent_data/mrc/mrc_add_interval/"

generate_data = list()

index = 1


def get_concrete(paragraphs, independents, dependents, s, e, title, interval_index):
    global index
    concrete = dict()
    concrete["context"] = paragraphs[s:e]
    concrete["qas"] = list()

    independ_index = 1
    for independ in independents:
        start, end = int(independ[2][0]), int(independ[2][1])
        if end - start <= 128 and start != end and start >= s and end <= e:
            question = "针对%s第%d个片段第%d个独立权利要求是？" % (title, interval_index, independ_index)
            index += 1
            independ_index += 1
            answer = paragraphs[start:end]
            answer_start = start - s
            id = str(index)

            answer = {"text": answer, "answer_start": answer_start}

            qa = dict()
            qa["question"] = question
            qa["answers"] = list()
            qa["answers"].append(answer)
            qa["id"] = id

            concrete["qas"].append(qa)

    depend_index = 1
    for dependent in dependents:
        start, end = int(dependent[2][0]), int(dependent[2][1])
        if end - start <= 128 and start != end and start >= s and end < e:
            question = "针对%s第%d片段第%d个从属权利要求是？" % (title, interval_index, depend_index)
            index += 1
            depend_index += 1
            answer = paragraphs[start:end]
            answer_start = start - s
            id = str(index)

            answer = {"text": answer, "answer_start": answer_start}

            qa = dict()
            qa["question"] = question
            qa["answers"] = list()
            qa["answers"].append(answer)
            qa["id"] = id

            concrete["qas"].append(qa)
    return concrete


def load_json(file_list):
    global generate_data
    for file_name in file_list:
        json_file = os.path.join(json_dir, file_name)
        with open(json_file) as fp:
            patent_json = json.load(fp)
            title = patent_json["title"]
            paragraphs = patent_json["invention_content"]
            independents = patent_json["independent"]
            dependents = patent_json["dependent"]

            context_len = len(paragraphs)

            if context_len > 2048:
                continue

            passage = dict()
            passage["title"] = title
            passage["paragraphs"] = list()

            interval = 512

            interval_index = 1
            for i in range(0, context_len, interval):
                start = i
                end = start + interval
                if end > context_len:
                    start = context_len - interval
                    end = context_len - 1
                    concrete = get_concrete(paragraphs, independents, dependents, start, end, title, interval_index)
                    # passage["paragraphs"].append(concrete)
                    break
                concrete = get_concrete(paragraphs, independents, dependents, start, end, title, interval_index)
                passage["paragraphs"].append(concrete)
                interval_index += 1
            generate_data.append(passage)


def generate_file(dev_per, test_per):
    global generate_data
    l = len(generate_data)
    train_per = 1 - dev_per - test_per
    random.shuffle(generate_data)
    print(l)

    if not os.path.exists(mrc_dir):
        os.makedirs(mrc_dir)

    with open(os.path.join(mrc_dir, "train_v1.1.json"), "w") as fp:
        source = dict()
        source["version"] = "1.1"
        source["data"] = generate_data[:int(l * train_per)]
        source_string = json.dumps(source)
        fp.write(source_string)

    with open(os.path.join(mrc_dir, "dev_v1.1.json"), "w") as fp:
        source = dict()
        source["version"] = "1.1"
        source["data"] = generate_data[int(l * train_per):int(l * (train_per + dev_per))]
        source_string = json.dumps(source)
        fp.write(source_string)

    with open(os.path.join(mrc_dir, "test_v1.1.json"), "w") as fp:
        source = dict()
        source["version"] = "1.1"
        source["data"] = generate_data[int(l * (train_per + dev_per)):]
        source_string = json.dumps(source)
        fp.write(source_string)


if __name__ == "__main__":
    file_list = os.listdir(json_dir)
    load_json(file_list)
    generate_file(0.05, 0.05)
