export SQUAD_DIR=../../patent_data/mrc_add_interval/
export CUDA_VISIBLE_DEVICES=0

python train.py \
    --model_name bertbidaf \
    --pretrain_model_path  ../pretrain_model/bert_model/ \
    --data_dir $SQUAD_DIR \
    --output_dir ../mrc_models/bidaf_model/ \
    --best_dir ../mrc_models/bidaf_model/best_model/ \
    --predict_dir ../mrc_models/bidaf_model/predict/ \
    --train_file train_v1.1.json \
    --dev_file dev_v1.1.json \
    --test_file test_v1.1.json \
    --max_seq_length 512 \
    --doc_stride 128 \
    --max_query_length 32 \
    --max_answer_length 128 \
    --learning_rate 2e-5 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_gpu_train_batch_size 6 \
    --per_gpu_eval_batch_size 6 \
    --num_train_epochs 10.0 \
    --logging_steps 5000 \
    --save_steps 20000 \
    --overwrite_output_dir