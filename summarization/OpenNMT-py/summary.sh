export CUDA_VISIBLE_DEVICES=0
python translate.py -src ../../../patent_data/summary/test.src \
                    -output ../test_result/reuse_copy_attn_loss_rnn_256_pred.txt \
                    -model ../model/reuse_copy_attn_loss_rnn_256/general_attention_step_30000.pt \
                    -beam_size 5 \
                    -replace_unk \
                    -max_length 512 \
                    -batch_size 10 \
                    -gpu 0



