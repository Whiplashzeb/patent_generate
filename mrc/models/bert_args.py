import argparse

def get_args():
    parser = argparse.ArgumentParser(description="base bert")

    parser.add_argument(
        "--pretrain_model_path",
        default=None,
        type=str,
        required=True,
        help="path to pretrained_model"
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="the output directory where the model checkpoints and predictions will be written"
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
        help="predict result to save"
    )

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="the input data dir"
    )

    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="the input training file"
    )

    parser.add_argument(
        "--dev_file",
        default=None,
        type=str,
        help="the dev file"
    )

    parser.add_argument(
        "--test_file",
        default=None,
        type=str,
        help="the test file"
    )

    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null."
    )

    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="the maximum total input sequence length after Wordpiece tokenizaiton."
        "longer than this will be truncated, and sequence shorter than this will be padded"
    )

    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks"
    )

    parser.add_argument(
        "--max_query_length",
        default=32,
        type=int,
        help="the maximum number of tokens for the question, Question longer than this will be truncated to this length"
    )

    parser.add_argument(
        "--max_answer_length",
        default=128,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another."
    )

    parser.add_argument(
        "--do_train",
        action="store_true",
        help="whether to run training."
    )

    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="whether to run evaluation."
    )

    parser.add_argument(
        "--do_test",
        action="store_true",
        help="whether to run test data."
    )

    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="run evaluation during training at each logging step"
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="batch size per gpu for training"
    )

    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="batch size per gpu for evaluation"
    )

    parser.add_argument(
        "--num_train_epochs",
        default=5.0,
        type=float,
        help="total number of training epochs to perform"
    )

    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="linear warmup over warmup_steps"
    )

    parser.add_argument(
        "--n_best_size",
        default=50,
        type=int,
        help="the total number of n-best predictions to generate in the nbest_predictions.json output file"
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=5000,
        help="log every X updates steps"
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=20000,
        help="save checkpoints every X updates steps"
    )

    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="the initial learning rate for Adam"
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass."
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
        help="Epsilon for Adam optimizer"
    )

    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm"
    )

    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="overwrite the content of the output dir"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for initialization"
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="multiple threads for converting example to features"
    )

    args = parser.parse_args()

    return args

