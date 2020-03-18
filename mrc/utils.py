import os
import random
import numpy as np
from tqdm import tqdm, trange
import timeit

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from transformers.data.processors.squad import SquadResult, SquadV1Processor
from transformers import squad_convert_examples_to_features, AdamW, get_linear_schedule_with_warmup
from transformers.data.metrics.squad_metrics import compute_predictions_log_probs, compute_predictions_logits, squad_evaluate


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def load_and_cache_examples(args, logger, tokenizer, data_type, output_examples):
    input_dir = args.data_dir
    cached_features_file = os.path.join(input_dir, "cached_{}".format(data_type))

    # init features and datasets from cache if it exists
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"]
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)
        processor = SquadV1Processor()

        if data_type == "train":
            examples = processor.get_train_examples(args.data_dir, args.train_file)
        elif data_type == "dev":
            examples = processor.get_dev_examples(args.data_dir, args.dev_file)
        elif data_type == "test":
            examples = processor.get_dev_examples(args.data_dir, args.test_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=(data_type == "train"),
            return_dataset="pt",
            threads=args.threads
        )

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset


def train(args, logger, train_dataset, model, tokenizer):
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

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

    logger.info("***** Running training *****")
    logger.info(" Num examples = %d", len(train_dataset))
    logger.info(" Num epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=False)
    set_seed(args)

    best_f1 = 0.0

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4]
            }

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
                results = evaluate(args, logger, model, tokenizer, "dev", "dev")
                for key, value in results.items():
                    tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logger.info("loss is {}".format((tr_loss - logging_loss) / args.logging_steps))
                logger.info("learning rate is {}".format(scheduler.get_lr()[0]))
                logging_loss = tr_loss
                for key, value in results.items():
                    logger.info("eval_{}: {}".format(key, value))
                if results["f1"] > best_f1:
                    best_f1 = results["f1"]
                    best_dir = args.best_dir
                    if not os.path.exists(best_dir):
                        os.makedirs(best_dir)

                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    torch.save(args, os.path.join(best_dir, "training_args.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(best_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(best_dir, "scheduler.pt"))

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

    tb_writer.close()
    return global_step, tr_loss / global_step, best_f1



def evaluate(args, logger, model, tokenizer, data_type, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, logger, tokenizer, data_type, output_examples=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # eval
    logger.info(" ***** Running evaluation {} *****".format(prefix))
    logger.info(" ***** Num examples = %d", len(dataset))
    logger.info(" Batch size = %d", args.eval_batch_size)

    all_results = list()
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Eval"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    eval_time = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", eval_time, eval_time / len(dataset))

    if not os.path.exists(args.predict_dir):
        os.makedirs(args.predict_dir)

    # Compute predictions
    output_prediction_file = os.path.join(args.predict_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.predict_dir, "nbest_predictions_{}.json".format(prefix))

    predcitions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        False,
        output_prediction_file,
        output_nbest_file,
        None,
        False,
        False,
        args.null_score_diff_threshold,
        tokenizer
    )

    results = squad_evaluate(examples, predcitions)
    return results
