export CUDA_VISIBLE_DEVICES=0
python train.py -data ../../../patent_data/summary/processed \
                -word_vec_size 128 \
                -encoder_type brnn \
                -enc_layers 1 \
                -dec_layers 1 \
                -enc_rnn_size 512 \
                -dec_rnn_size 512 \
                -dropout 0.3 \
                -rnn_type LSTM \
                -global_attention general \
                -copy_attn \
                -copy_attn_type general \
                -bridge \
                -batch_size 32 \
                -valid_batch_size 32 \
                -train_steps 30000 \
                -valid_steps 5000 \
                -optim adam \
                -learning_rate 0.001 \
                -save_model ../model/copy_attn_general_bridge_512/general_attention \
                -save_checkpoint_steps 5000 \
                -gpu_ranks 0 \
                -seed 777 \

# -copy_loss_by_seqlength \
# -coverage_attn \
# -reuse_copy_attn \