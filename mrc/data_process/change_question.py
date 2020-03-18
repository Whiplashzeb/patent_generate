import os
import json

mrc_dir = "../../patent_data/mrc/mrc_add_title/"
change_dir = "../../patent_data/mrc/mrc_add_interval/"

def load_json(file_name):
    mrc_file = os.path.join(mrc_dir, file_name)
    with open(mrc_file) as fp:
        mrc_json = json.load(fp)

        passages = mrc_json["data"]
        for passage in passages:
            paragraphs = passage["paragraphs"]
            for i, concrete in enumerate(paragraphs):
                pass