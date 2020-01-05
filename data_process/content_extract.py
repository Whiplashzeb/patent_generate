import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from data_process.patent import PatentDes, PatentClaim

claim_dir = "../../patent_data/claim/"
des_dir = "../../patent_data/des/"
claim_json_dir = "../../patent_data/claim_json/"
des_json_dir = "../../patent_data/des_json/"


def create_json(file_list):
    for file_name in file_list:
        des_file = os.path.join(des_dir, file_name)
        des = PatentDes(des_file)
        des_json = des.get_json()

        claim_file = os.path.join(claim_dir, file_name)
        claim = PatentClaim(claim_file)
        claim_json = claim.get_json()

        number = des_json["number"]
        if des_json["title"] != "" and des_json["invention_content"] != "" and claim_json["independent"] != []:
            json_file_name = "%s.json" % number

            des_json_file = os.path.join(des_json_dir, json_file_name)
            with open(des_json_file, "w") as fp:
                des_string = json.dumps(des_json)
                fp.write(des_string)

            claim_json_file = os.path.join(claim_json_dir, json_file_name)
            with open(claim_json_file, "w") as fp:
                claim_string = json.dumps(claim_json)
                fp.write(claim_string)


if __name__ == "__main__":
    file_list = os.listdir(des_dir)
    with ProcessPoolExecutor(48) as exe:
        i = 0
        fs = []
        while i < len(file_list):
            fs.append(exe.submit(create_json, file_list[i:i + 2000]))
            i += 2000
        for future in as_completed(fs):
            result = future.result()
