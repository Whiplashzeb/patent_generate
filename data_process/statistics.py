import os
import json

json_dir = "../../patent_data/json/"

inventions_len = [0 for _ in range(100)]  # 按照500的范围进行统计
independents_len = [0 for _ in range(100)]  # 按照100的范围进行统计
dependents_len = [0 for _ in range(100)]  # 按照100的范围进行统计
match_file_num = 0  # 统计能匹配到的文件个数
match_independents = 0  # 统计能匹配到的独立权利要求个数
match_dependents = 0  # 统计能匹配到的从属权利要求个数
unmatch_independents = 0  # 统计匹配不到的独立权利要求个数
unmatch_dependents = 0  # 统计匹配不到的从属权利要求个数


def load_json(file_list):
    global inventions_len
    global independents_len
    global dependents_len
    global match_file_num
    global match_independents
    global match_dependents
    global unmatch_independents
    global unmatch_dependents

    for file_name in file_list:
        json_file = os.path.join(json_dir, file_name)
        with open(json_file) as fp:
            match_flag = False

            patent = json.load(fp)
            invention_content = patent["invention_content"]
            independent = patent["independent"]
            dependent = patent["dependent"]

            content_len = len(invention_content)
            if content_len // 500 < 100:
                inventions_len[content_len // 500] += 1

            for independ in independent:
                character, (start, end) = independ[1], (int(independ[2][0]), int(independ[2][1]))
                independ_len = len(character)
                if independ_len // 100 < 100:
                    independents_len[independ_len // 100] += 1
                if end - start != 0:
                    match_flag = True
                    match_independents += 1
                else:
                    unmatch_independents += 1

            for depend in dependent:
                limited, (start, end) = depend[1], (int(depend[2][0]), int(depend[2][1]))
                depend_len = len(limited)
                if depend_len // 100 < 100:
                    dependents_len[depend_len // 100] += 1
                if end - start != 0:
                    match_flag = True
                    match_dependents += 1
                else:
                    unmatch_dependents += 1

            if match_flag:
                match_file_num += 1


def summary():
    global inventions_len
    global independents_len
    global dependents_len
    global match_file_num
    global match_independents
    global match_dependents
    global unmatch_independents
    global unmatch_dependents

    with open("summary.txt", "w") as fp:
        fp.write("发明内容长度统计:" + '\n')
        fp.write(" ".join(map(str, inventions_len)))
        fp.write("\n独立权利要求长度统计:" + "\n")
        fp.write(" ".join(map(str, independents_len)))
        fp.write("\n从属权利要求长度统计:" + "\n")
        fp.write(" ".join(map(str, dependents_len)))
        fp.write("\n可匹配文件个数:" + str(match_file_num) + "\n")
        fp.write("可匹配独立权利要求个数:" + str(match_independents) + "\n")
        fp.write("可匹配从属权利要求个数:" + str(match_dependents) + "\n")
        fp.write("不可匹配独立权利要求个数:" + str(unmatch_independents) + "\n")
        fp.write("不可匹配从属权利要求个数:" + str(unmatch_dependents) + "\n")


if __name__ == "__main__":
    file_list = os.listdir(json_dir)
    load_json(file_list)
    summary()