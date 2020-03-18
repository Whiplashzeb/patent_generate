python preprocess.py -train_src ../../../patent_data/summary/train.src \
                     -train_tgt ../../../patent_data/summary/train.tgt \
                     -valid_src ../../../patent_data/summary/dev.src \
                     -valid_tgt ../../../patent_data/summary/dev.tgt \
                     -save_data ../../../patent_data/summary/processed \
                     -src_seq_length 1024 \
                     -tgt_seq_length 512 \
                     -dynamic_dict \
                     -share_vocab \
                     -shard_size 50000