import os
import csv
import random

train_dir = "../../../patent_data/classify/train.tsv"
change_dir = "../../../patent_data/classify_11/train.tsv"

if __name__ == "__main__":
    all = 0
    train_1 = list()
    train_0 = list()
    train_2 = list()
    with open(train_dir,  "r", encoding="utf-8-sig") as fp:
        all_train = list(csv.reader(fp, delimiter="\t"))
        all = len(all_train)
        for line in all_train:
            label, text_a, text_b = line[0], line[3], line[4]
            if label == "1":
                train_1.append([label,text_a,text_b])
            if label == "0":
                train_0.append([label,text_a,text_b])
            if label == "-1":
                train_2.append([label,text_a,text_b])

    filter = train_1 + train_2
    for line in train_0:
        if random.randint(1,10) in [1,2,3]:
            filter.append(line)
    random.shuffle(filter)
    print(len(filter))
    with open(change_dir, "w") as fp:
        fp.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for data in filter:
            fp.write("%s\t1\t1\t%s\t%s\n" % (data[0], data[1], data[2]))