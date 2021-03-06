export CUDA_VISIBLE_DEVICES=0
python train.py -data ../../../patent_data/summary/processed \
                -layers 6 \
                -rnn_size 128 \
                -word_vec_size 128 \
                -transformer_ff 2048 \
                -heads 8 \
                -encoder_type transformer \
                -decoder_type transformer \
                -position_encoding \
                -train_steps 100000 \
                -dropout 0.2 \
                -batch_size 4096 \
                -batch_type tokens \
                -normalization tokens \
                -optim adam \
                -adam_beta2 0.998 \
                -learning_rate 0.001 \
                -valid_steps 5000 \
                -save_checkpoint_steps 5000 \
                -save_model ../model/transformer/transformer \
                -gpu_ranks 0 \
                -seed 777