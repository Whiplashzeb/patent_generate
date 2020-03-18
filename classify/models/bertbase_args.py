import argparse

def get_parse():
    parser = argparse.ArgumentParser(description="bert base")

    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        required=True,
        help="to choose model"
    )

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="the input data dir, should contain the .tsv files for the task"
    )

    parser.add_argument(
        "--pretrain_model_path",
        default=None,
        type=str,
        required=True,
        help="path to pretrained model"
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="the output directory where the model checkpoint will be written"
    )

    parser.add_argument(
        "--best_dir",
        default=None,
        type=str,
        required=True,
        help="the eval f1 best model to save"
    )
    
    parser.add_argument(
        "--predict_dir",
        default=None,
        type=str,
        required=True,
        help="save some predict files"
    )

    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded."
    )

    parser.add_argument(
        "--do_train",
        action="store_true",
        help="whether to run training"
    )

    parser.add_argument(
        "--do_test",
        action="store_true",
        help="whether to run test"
    )

    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="run evaluation during training at each logging step"
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=16,
        type=int,
        help="batch size per GPU for training"
    )

    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=16,
        type=int,
        help="batch size per GPU for evaluating"
    )

    parser.add_argument(
        "--num_train_epochs",
        default=8.0,
        type=float,
        help="total number of training epochs to perform"
    )

    parser.add_argument(
        "--logging_steps",
        default=5000,
        type=int,
        help="log every X updates steps"
    )

    parser.add_argument(
        "--save_steps",
        default=20000,
        type=int,
        help="save checkpoint every X updates steps"
    )

    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="overwrite the content of the output directory"
    )
    
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="overwrite exist cache file"
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="random seed for initialization"
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="number of updates steps to accumulate before performing a backward pass"
    )

    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="the initial learning rate for Adam"
    )

    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="weight decay if we apply some"
    )

    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for adam optimizer"
    )

    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm"
    )

    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="linear warmup over warmup_steps"
    )

    args = parser.parse_args()

    return args

