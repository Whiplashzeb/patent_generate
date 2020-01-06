import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import chain
from tqdm import tqdm

claim_json_dir = "../../patent_data/claim_json/"
des_json_dir = "../../patent_data/des_json/"
align_json_dir = "../../patent_data/align/"

def cut_sents(para: str):
    """
    断句，如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放在双引号后
    """
    total_len = len(para)
    para = re.sub(r'([：；，､\u3000、﹔·！？｡。])([^”’])', r"\1@@sub@@\2", para)  # 单字符断句符
    para = re.sub(r'(\.{6})([^”’])', r"\1@@sub@@\2", para)  # 英文省略号
    para = re.sub(r'(…{2})([^”’])', r"\1@@sub@@\2", para)  # 中文省略号
    para = re.sub(r'([：；，､\u3000、﹔·！？｡。][”’])([^，。！？?])', r'\1@@sub@@\2', para)

    span = list()
    prev = 0

    index = para.find("@@sub@@")
    while index > 0:
        if index > 3:
            span.append((prev, prev + index))
        prev += index
        para = para[index + 7:]
        index = para.find("@@sub@@")
    if len(para) > 3:
        span.append((prev, prev + len(para)))
    assert len(span) == 0 or span[-1][-1] <= total_len

    return span


def lcs(str_a, str_b):
    """
    寻找两个字符串间的最长公共子序列
    """
    if len(str_a) == 0 or len(str_b) == 0:
        return 0
    dp = [[0 for _ in range(len(str_b) + 1)] for _ in range(len(str_a) + 1)]
    for i in range(1, len(str_a) + 1):
        for j in range(1, len(str_b) + 1):
            if str_a[i - 1] == str_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max([dp[i - 1][j], dp[i][j - 1]])

    return dp[len(str_a)][len(str_b)]


def worker(data_jsons: list, lcs_threshold=0.8):
    for des_json, claim_json in data_jsons:
        alignment = dict()
        alignment.update(des_json)
        alignment.update(claim_json)

        invention_content = des_json["invention_content"]
        des_str = invention_content
        des_spans = cut_sents(des_str)

        for claim_type in ("independent, dependent"):
            for claim_id, (_, claim_str) in enumerate(claim_json[claim_type]):
                claim_spans = cut_sents(claim_str)
                if len(des_spans) == 0 or len(claim_spans) == 0:
                    alignment[claim_type][claim_id].append((-1, -1))
                    continue
                # 计算一个文书内文本分段间的lcs匹配率,lcs_rates中保存匹配矩阵
                lcs_rates = list()
                for i in range(len(claim_spans)):
                    lcs_rates.append([])
                    for j in range(len(des_spans)):
                        des_span = des_str[des_spans[j][0]:des_spans[j][1]]
                        claim_span = claim_str[claim_spans[i][0]:claim_spans[i][1]]
                        ij_lcs = lcs(claim_span, des_span)
                        l1 = len(des_span)
                        l2 = len(claim_span)
                        l = min(l1, l2)
                        assert l > 0
                        lcs_rate = ij_lcs / l
                        if lcs_rate >= lcs_threshold:
                            lcs_rates[i].append(1)
                        else:
                            lcs_rates[i].append(-1)
                # 从后到前寻找最长匹配，贪心方案
                matched_des = list()
                border = len(des_spans) - 1
                for i in range(len(claim_spans) - 1, -1, -1):
                    for j in range(border, -1, -1):
                        if lcs_rates[i][j] > 0:
                            matched_des.append((i, j))
                            border = j - 1
                            break
                if len(matched_des) > 0:
                    matched_range = (des_spans[matched_des[-1][1]][0], des_spans[matched_des[0][1]][1])
                else:
                    matched_range = (-1, -1)
                alignment[claim_type][claim_id].append(matched_range)

        align_file_name = "%s.json" % alignment["number"]
        align_json_file = os.path.join(align_json_dir, align_file_name)
        with open(align_json_file, "w") as fp:
            align_string = json.dumps(alignment)
            fp.write(align_string)



if __name__ == "__main__":
    file_list = os.listdir(des_json_dir)
    data_jsons = list()
    for file_name in tqdm(file_list):
        with open(os.path.join(des_json_dir, file_name)) as des_file, open(os.path.join(claim_json_dir, file_name)) as claim_file:
            des_json = json.load(des_file)
            claim_json = json.load(claim_file)
            assert claim_json["number"] == des_json["number"]
            data_jsons.append((des_json, claim_json))
    print(len(data_jsons))

    process_count = 48
    with ProcessPoolExecutor(process_count) as executor:
        shard_size = len(data_jsons) // process_count + 1
        i = 0
        fs = []
        while i < len(data_jsons):
            fs.append(executor.submit(worker, data_jsons[i:i + shard_size]))
            i += shard_size
        for future in as_completed(fs):
            result = future.result()