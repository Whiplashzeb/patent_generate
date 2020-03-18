from models.BertBase import BertBase
from models.BertLSTM import BertLSTM
from models.BertATT import BertATT
from models.BertCNN import BertCNN
from models.BertRCNN import BertRCNN
from models.BertDPCNN import BertDPCNN
from models.bertbase_args import get_parse
from data_process.data_processor import ClassifyProcessor
from utils import set_seed, load_and_cache_examples, train

import logging
import os

import torch

from transformers import BertConfig, BertTokenizer

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

    # prepare task processor
    processor = ClassifyProcessor()
    args.output_mode = "classification"
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # load pretrained model and tokenizer
    config = BertConfig.from_pretrained(args.pretrain_model_path, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path, do_lower_case=False)
    if args.model_name == "bertbase":
        model = BertBase.from_pretrained(args.pretrain_model_path, config=config)
    elif args.model_name == "bertlstm":
        model = BertLSTM.from_pretrained(args.pretrain_model_path, config=config, rnn_hidden_size=512)
    elif args.model_name == "bertatt":
        model = BertATT.from_pretrained(args.pretrain_model_path, config=config)
    elif args.model_name == "bertcnn":
        model = BertCNN.from_pretrained(args.pretrain_model_path, config=config, n_filters=256, filter_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    elif args.model_name == "bertrcnn":
        model = BertRCNN.from_pretrained(args.pretrain_model_path, config=config, rnn_hidden_size=512, layers=2, dropout=0.2)
    elif args.model_name == "bertdpcnn":
        model = BertDPCNN.from_pretrained(args.pretrain_model_path, config=config, filter_num=256)
    model.to(args.device)

    logger.info("Training/Evaluation parameters %s", args)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, file_mode="train")
        global_step, tr_loss, best_f1 = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s, best_f1 = %s", global_step, tr_loss, best_f1)
