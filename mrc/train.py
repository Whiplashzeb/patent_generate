from models.BertMRC import BertMRC
from models.BertBiDAF import BertBiDAF
from models import bert_args
from utils import load_and_cache_examples, train, set_seed
import logging
import os

import torch

from transformers import BertConfig, BertTokenizer

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = bert_args.get_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # set logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # set seed
    set_seed(args)

    # load pretrain model and tokenizer
    config = BertConfig.from_pretrained(args.pretrain_model_path)
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path, do_lower_case=False)
    if args.model_name == "bertmrc":
        model = BertMRC.from_pretrained(args.pretrain_model_path, config=config)
    elif args.model_name == "bertbidaf":
        model = BertBiDAF.from_pretrained(args.pretrain_model_path, config=config, rnn_hidden_size=128, dropout=0.2)
    model.to(args.device)

    logger.info("Training parameters $s", args)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, logger, tokenizer, "train", False)
        global_step, tr_loss, best_f1 = train(args, logger, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s, best f1 = %s", global_step, tr_loss, best_f1)
