import os
import json
from patent import Patent
from concurrent.futures import ProcessPoolExecutor, as_completed

claim_dir = "../../patent_data/claim/"
des_dir = "../../patent_data/des/"
json_dir = "../../patent_data/json/"

def create_json(file_list):
    for file_name in file_list:
        des_file = os.path.join(des_dir, file_name)
        claim_file = os.path.join(claim_dir, file_name)

        patent = Patent(des_file, claim_file, 0.8)
        patent_json = patent.get_json()

        number = patent_json["number"]
        if patent_json["title"] == "" or patent_json["invention_content"] == "" or patent_json["independent"] == []:
            print(number)
            continue

        json_file_name = "%s.json" % number
        json_file = os.path.join(json_dir, json_file_name)
        with open(json_file, "w") as fp:
            patent_string = json.dumps(patent_json)
            fp.write(patent_string)



if __name__ == "__main__":
    file_list = os.listdir(des_dir)

    process_count = 48
    with ProcessPoolExecutor(process_count) as executor:
        shard_size = len(file_list) // process_count + 1
        i = 0
        fs = []
        while i < len(file_list):
            fs.append(executor.submit(create_json, file_list[i:i+shard_size]))
            i += shard_size
        for future in as_completed(fs):
            result = future.result()
