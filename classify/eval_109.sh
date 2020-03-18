export GLUE_DIR=../../patent_data/classify/augment_data
export CUDA_VISIBLE_DEVICES=0

python eval.py \
      --model_name bertcnn \
      --data_dir $GLUE_DIR/ \
      --pretrain_model_path  ../pretrain_model/bert-base-chinese/ \
      --output_dir ./checkpoint/cnn_model/ \
      --best_dir ./checkpoint/cnn_model/best_model/ \
      --predict_dir ./checkpoint/cnn_model/predict/ \
      --max_seq_length 128 \
      --do_test \
      --evaluate_during_training \
      --per_gpu_train_batch_size 32 \
      --per_gpu_eval_batch_size 32 \
      --num_train_epochs 8.0 \
      --logging_steps 5000 \
      --save_steps 25000 \
      --learning_rate 2e-5 \
      --overwrite_output_dir
