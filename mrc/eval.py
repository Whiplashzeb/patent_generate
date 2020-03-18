from models.BertMRC import BertMRC
from models.bert_args import get_args
from utils import evaluate, set_seed
import logging
import os

import torch

from transformers import BertConfig, BertTokenizer

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = get_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

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
    # config = BertConfig.from_pretrained(args.best_dir)
    tokenizer = BertTokenizer.from_pretrained(args.best_dir, do_lower_case=False)
    model = BertMRC.from_pretrained(args.best_dir)
    model.to(args.device)

    if args.do_test:
        result = evaluate(args, logger, model, tokenizer, data_type="test", prefix="test")
        # result = dict((k, v) for k, v in result.items())
        for key, value in result.items():
            logger.info("eval_{}: {}".format(key, value))
