from data_process.data_processor import ClassifyProcessor

import os
import random
import json
import numpy as np
import logging
from tqdm import tqdm, trange
from sklearn import metrics

import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def load_and_cache_examples(args, tokenizer, file_mode):
    processor = ClassifyProcessor()
    output_mode = "classification"

    cached_features_file = os.path.join(args.data_dir, "cached_{}".format(file_mode))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("loading features from cached file {}".format(cached_features_file))
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at {}".format(cached_features_file))
        label_list = processor.get_labels()

        if file_mode == "train":
            examples = (processor.get_train_examples(args.data_dir))
        elif file_mode == "dev":
            examples = (processor.get_dev_examples(args.data_dir))
        else:
            examples = (processor.get_test_examples(args.data_dir))

        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0
        )

        logger.info("Saving features into cached file {}".format(cached_features_file))
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def compute_metrics(preds, out_label_ids):
    every_label_precision = metrics.precision_score(out_label_ids, preds, labels=[0, 1, 2], average=None)
    every_label_recall = metrics.recall_score(out_label_ids, preds, labels=[0, 1, 2], average=None)
    every_label_f1 = metrics.f1_score(out_label_ids, preds, labels=[0, 1, 2], average=None)

    macro_precision = metrics.precision_score(out_label_ids, preds, average="macro")
    micro_precision = metrics.precision_score(out_label_ids, preds, average="micro")
    weighted_precision = metrics.precision_score(out_label_ids, preds, average="weighted")
    macro_recall = metrics.recall_score(out_label_ids, preds, average="macro")
    micro_recall = metrics.recall_score(out_label_ids, preds, average="micro")
    weighted_recall = metrics.recall_score(out_label_ids, preds, average="weighted")
    macro_f1 = metrics.f1_score(out_label_ids, preds, average='macro')
    micro_f1 = metrics.f1_score(out_label_ids, preds, average='micro')
    weighted_f1 = metrics.f1_score(out_label_ids, preds, average='weighted')

    return {
        "every_label_precision": every_label_precision,
        "every_label_recall": every_label_recall,
        "every_label_f1": every_label_f1,
        "macro_precision": macro_precision,
        "micro_precision": micro_precision,
        "weighted_precision": weighted_precision,
        "macro_recall": macro_recall,
        "micro_recall": micro_recall,
        "weighted_recall": weighted_recall,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
    }


def train(args, train_dataset, model, tokenizer):
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if os.path.isfile(os.path.join(args.pretrain_model_path, "optimizer.pt")) and \
            os.path.isfile(os.path.join(args.pretrain_model_path, "scheduler.pt")):
        optimizer.load_state_dict(torch.load(os.path.join(args.pretrain_model_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.pretrain_model_path, "scheduler.pt")))

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=False)
    set_seed(args)

    best_f1 = 0

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
            outputs = model(**inputs)

            loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logs = dict()
                if args.evaluate_during_training:
                    results = evaluate(args, model, tokenizer, "dev", prefix="dev")
                    for key, value in results.items():
                        if key in ["every_label_precision", "every_label_recall", "every_label_f1"]:
                            continue
                        eval_key = "eval_{}".format(key)
                        logs[eval_key] = value

                    f1 = results["weighted_f1"]
                    if f1 > best_f1:
                        best_f1 = f1
                        output_dir = args.best_dir

                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (model.module if hasattr(model, "module") else model)
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                learning_rate_scalar = scheduler.get_lr()[0]
                logs["learning_rate"] = learning_rate_scalar
                logs["loss"] = loss_scalar
                logging_loss = tr_loss
                for key, value in logs.items():
                    tb_writer.add_scalar(key, value, global_step)
                print(json.dumps({**logs, **{"step": global_step}}))

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, "checkpoints-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (model.module if hasattr(model, "module") else model)
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)
    tb_writer.close()
    return global_step, tr_loss / global_step, best_f1


def evaluate(args, model, tokenizer, data_type, prefix=""):
    eval_outputs_dir = args.predict_dir
    if not os.path.exists(eval_outputs_dir):
        os.makedirs(eval_outputs_dir)

    results = dict()

    eval_dataset = load_and_cache_examples(args, tokenizer, data_type)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)

    # logger.info(("%s", str(preds[:100])))

    result = compute_metrics(preds, out_label_ids)
    results.update(result)

    output_eval_file = os.path.join(eval_outputs_dir, "{}_results.txt".format(prefix))
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return results
