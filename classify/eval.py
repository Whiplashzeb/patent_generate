from models.BertBase import BertBase
from models.BertLSTM import BertLSTM
from models.BertCNN import BertCNN
from models.BertATT import BertATT
from models.BertDPCNN import BertDPCNN
from models.BertRCNN import BertRCNN
from models.bertbase_args import get_parse
from utils import set_seed, evaluate

import logging
import os

import torch

from transformers import BertTokenizer

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = get_parse()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("output directory already exists and is not empty")

    # set cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # set seed
    set_seed(args)

    # load pretrain model and tokenizer
    # config = BertConfig.from_pretrained(args.best_dir)
    tokenizer = BertTokenizer.from_pretrained(args.best_dir, do_lower_case=False)
    if args.model_name == "bertbase":
        model = BertBase.from_pretrained(args.best_dir)
    elif args.model_name == "bertlstm":
        model = BertLSTM.from_pretrained(args.best_dir, rnn_hidden_size=512)
    elif args.model_name == "bertcnn":
        model = BertCNN.from_pretrained(args.best_dir, n_filters=256, filter_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    elif args.model_name == "bertatt":
        model = BertATT.from_pretrained(args.best_dir)
    elif args.model_name == "bertrcnn":
        model = BertRCNN.from_pretrained(args.best_dir, rnn_hidden_size=512, layers=2, dropout=0.2)
    elif args.model_name == "bertdpcnn":
        model = BertDPCNN.from_pretrained(args.best_dir, filter_num=256)
    model.to(args.device)

    logger.info("Training/Evaluation parameters %s", args)

    if args.do_test:
        result = evaluate(args, model, tokenizer, data_type="test", prefix="test")
        for key, value in result.items():
            logger.info("eval_{}: {}".format(key, value))
