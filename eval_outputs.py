"""
This module implements evaluation functions for sft and policy models.
It uses the same generation config as used in policy rolling out.
A detailed csv as well as an overview of the results will be saved.
"""

import argparse
import csv
import heapq
import json
import os
import pickle
from typing import Dict, List

import evaluate
import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from sacrebleu.metrics import BLEU
from sacremoses import MosesTokenizer
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, AutoModelForSeq2SeqLM)

from utils import (LONG_T5_XL, GEMMA_2B, LLAMA3_8B, MAX_OUTPUT_LENGTHS,
                   OLMO_1B, PHI2_3B, SEED, VOA1500,
                   WORD_ACCESSIBILITY_MODEL, WORD_FREQ_CSV, build_sass_dataset,
                   compute_ari, compute_flesch_kincaid, compute_sent_len,
                   compute_token_accessibility, read_token_frequencies)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
metric_bleu = BLEU()
metric_sari = evaluate.load("sari")
metric_rouge = evaluate.load("rouge")
metric_bertscore = evaluate.load("bertscore")
# get word frequencies and the model to predict relative rare word's accessibility
token_freq = read_token_frequencies(WORD_FREQ_CSV)
top_100k_tokens = heapq.nlargest(100000, token_freq, key=token_freq.get)
# load for making predictions word accessibility
wa_model = pickle.load(open(WORD_ACCESSIBILITY_MODEL, "rb"))
total_tokens = sum(token_freq.values())
mt = MosesTokenizer(lang="en")
# VOA Word Book, Section A-Z, Science programs, and Organs of the body (1517 in total)
# from https://simple.wikipedia.org/wiki/Wikipedia:VOA_Special_English_Word_Book
# scraped on May 15, 2024
voa1500 = json.load(open(VOA1500, "r", encoding="utf-8"))


def calculate_metrics(
        generated_text: str, target_text: str, source_text: str
) -> Dict[str, float]:
    metrics_dict = {}
    generated_texts = [generated_text.strip()]
    source_texts = [source_text.strip()]
    target_texts = [[target_text.strip()]]
    metrics_dict.update({"ari": compute_ari(generated_texts[0])})
    metrics_dict.update({"fk": compute_flesch_kincaid(generated_texts[0])})
    metrics_dict.update(
        {"bleu": metric_bleu.corpus_score(generated_texts, target_texts).score}
    )
    metrics_dict.update(
        metric_sari.compute(
            sources=source_texts, predictions=generated_texts, references=target_texts
        )
    )
    _rouge = metric_rouge.compute(predictions=generated_texts, references=target_texts)
    metrics_dict.update({"rougeL": _rouge["rougeL"]})
    bertscore_result = metric_bertscore.compute(
        predictions=generated_texts,
        references=target_texts,
        lang="en",
        device="cpu",
        model_type="bert-large-uncased",
    )
    metrics_dict.update({"bertscore": np.mean(bertscore_result["f1"])})
    # complexity measure
    word_accessibility_list = []
    sent_len_list = []
    num_words = 0
    num_chars = 0
    num_voa_words = 0
    sents = sent_tokenize(generated_text)
    for sent in sents:
        sent_len_list.append(compute_sent_len(sent))
        for token in mt.tokenize(sent, escape=False):
            num_words += 1
            num_chars += len(token)
            if token.lower() in voa1500:
                num_voa_words += 1
            word_accessibility_list.append(
                compute_token_accessibility(
                    token, top_100k_tokens, wa_model, total_tokens, token_freq
                )
            )
    p = (num_voa_words / num_words) + 1e-12
    metrics_dict.update({"voa_log_ratio": np.log(p / (1 - p))})
    metrics_dict.update({"avg_sent_len": np.mean(sent_len_list)})
    metrics_dict.update({"avg_word_accessibility": np.mean(word_accessibility_list)})
    metrics_dict.update({"num_sents": len(sents)})
    metrics_dict.update({"avg_word_len": num_chars / num_words})
    return metrics_dict


def evaluate_model(
        model, dataset, tokenizer, generation_config, batch_size, model_type='clm',
        verbose=False
) -> List[Dict]:
    results = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch_samples = dataset[i: i + batch_size]
            # it is good to retokenize the ['query'] column for batch processing
            input_ids = torch.tensor(batch_samples["query_token"]).to(device)
            generated_tokens = model.generate(
                input_ids=input_ids, generation_config=generation_config
            )
            # only newly generated text are returned
            if model_type == 'clm':
                generated_texts = tokenizer.batch_decode(
                    generated_tokens[:, input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
            elif model_type == 'seq2seq':
                generated_texts = tokenizer.batch_decode(generated_tokens,
                                                         skip_special_tokens=True)
            for j, generated_text in enumerate(generated_texts):
                generated_text = generated_text.strip()
                result = calculate_metrics(
                    generated_text,
                    batch_samples["response"][j],
                    batch_samples["source"][j],
                )
                if verbose:
                    print(f'{generated_text=}')
                results.append(result | {"generated_text": generated_text})
    return results


if __name__ == "__main__":
    print("*" * 90)
    parser = argparse.ArgumentParser(
        description="Evaluate SFT and policy model outputs given model type and "
                    "validation ARI"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gemma-2b", "gemma-7b", "olmo-1b", "llama3-8b", "gpt2-xl", "phi-2",
                 'long-t5-tglobal-xl'],
        help="The model type (across runs) to evaluate",
    )
    parser.add_argument(
        "--eval_ppo",
        action='store_true',
        help="Flag to evaluate PPO model outputs. Defaults to False.",
    )
    parser.add_argument(
        "--upper_ari_bound",
        type=float,
        default=15.0,
        help="The upper bound of evaluation ARI for a checkpoint to be considered in "
             "the evaluation",
    )
    parser.add_argument(
        "--lower_ari_bound",
        type=float,
        default=8.0,
        help="The lower bound of evaluation ARI for a checkpoint to be considered in "
             "the evaluation",
    )
    parser.add_argument(
        "--reward", type=str, default="uam", choices=["uam", "ari"], help="Reward type"
    )
    parser.add_argument(
        "--batch_size", type=int, default=20, help="Batch size for inference"
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="Sampling top_p"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.01, help="Sampling temperature"
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help="Flag to print generated texts during evaluation. Defaults to False.",
    )
    args = parser.parse_args()
    torch.manual_seed(SEED)
    save_dir = f"eval_results_temp_{args.temperature}"
    os.makedirs(save_dir, exist_ok=True)

    print(
        f"Starting evaluation: only newly added runs whose checkpoints met "
        f"{args.lower_ari_bound} <= validation ARI <= {args.upper_ari_bound} "
        f"will be evaluated"
    )

    # identify the base model based on the provided model type argument
    if "gemma-2b" in args.model.lower():
        base_model = GEMMA_2B
    elif "olmo-1b" in args.model.lower():
        base_model = OLMO_1B
    elif "phi-2" in args.model.lower():
        base_model = PHI2_3B
    elif "llama3-8b" in args.model.lower():
        base_model = LLAMA3_8B
    elif "long-t5-tglobal-xl" in args.model.lower():
        base_model = LONG_T5_XL
    else:
        raise ValueError(f"Unknown model name {args.model}")

    # define the generation configuration
    test_generation_config = GenerationConfig(
        max_new_tokens=MAX_OUTPUT_LENGTHS[args.model.split('/')[-1].lower()],
        temperature=args.temperature + 1e-7,
        top_k=0.0,
        top_p=args.top_p,
        do_sample=True,
        num_return_sequences=1,
    )
    print(f"{test_generation_config=}")

    # load the overview file if it exists
    overview_path = os.path.join(save_dir, "overview.jsonl")
    if os.path.exists(overview_path):
        with open(overview_path, mode="r", encoding="utf-8") as f:
            overview = [json.loads(line) for line in f]
    else:
        overview = []
    evaluated_runs = {entry["run_path"] for entry in overview}

    # check and evaluate SFT models
    # SFT runs have slightly different naming conventions
    sft_base_model = base_model.split("/")[-1]
    sft_run_dir = os.path.join("ckpts", f"sft_{sft_base_model}")
    sft_checkpoints = os.listdir(sft_run_dir)
    if len(sft_checkpoints) != 1:
        raise ValueError(
            f"Expected exactly one checkpoint in {sft_run_dir}, but "
            f"found {len(sft_checkpoints)}."
        )
    sft_checkpoint = sft_checkpoints[0]
    sft_model_path = os.path.join(sft_run_dir, sft_checkpoint)
    if sft_run_dir not in evaluated_runs:
        print(f"Starting evaluation for {sft_model_path}")

    # load dataset and tokenizer
    # dataset = build_sass_dataset(sft_model_path, base_model, 'left')
    if 'long-t5' not in args.model:
        dataset = build_sass_dataset(sft_model_path, base_model, 'left')
    else:
        dataset = build_sass_dataset(sft_model_path, base_model, 'right')

    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)  # use the saved tokenizer
    if args.model in ['gemma-2b', 'olmo-1b', 'phi-2']:
        model = AutoModelForCausalLM.from_pretrained(
            sft_model_path, torch_dtype=torch.bfloat16
        )
        if args.model == 'phi-2':
            # resize embedding size for loading peft model
            model.resize_token_embeddings(len(tokenizer))
    elif args.model == 'llama3-8b':
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA3_8B, torch_dtype=torch.bfloat16
        )
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        # resize embedding size for loading peft model
        model.resize_token_embeddings(len(tokenizer))
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, sft_model_path)

    elif args.model == 'long-t5-tglobal-xl':
        model = AutoModelForSeq2SeqLM.from_pretrained(
            sft_model_path, torch_dtype=torch.bfloat16
        )
    else:
        raise RuntimeError(f"Illegal {args.model}.")

    model.to(device)

    # evaluate with test generation config
    eval_results = evaluate_model(
        model,
        dataset["test"],
        tokenizer,
        test_generation_config,
        batch_size=args.batch_size,
        model_type='clm' if args.model != 'long-t5-tglobal-xl' else 'seq2seq',
        verbose=args.verbose
    )

    # save evaluation results to CSV
    file_path = os.path.join(save_dir, f"{sft_model_path.replace('/', '|')}.csv")
    with open(file_path, mode="w", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=eval_results[0].keys())
        writer.writeheader()
        writer.writerows(eval_results)

    # calculate average and standard deviation of scores
    avg_scores = {
        f"avg_{metric}": np.mean([x[metric] for x in eval_results])
        for metric in eval_results[0].keys()
        if metric not in ["generated_text"]
    }
    std_scores = {
        f"std_{metric}": np.std([x[metric] for x in eval_results])
        for metric in eval_results[0].keys()
        if metric not in ["generated_text"]
    }

    # save the overview in JSONL format
    with open(overview_path, mode="a", encoding="utf-8") as f:
        json.dump(
            {"run_path": sft_run_dir}
            | {"ckpt_path": sft_model_path}
            | avg_scores
            | std_scores,
            f,
        )
        f.write("\n")

    # print out results
    print("*" * 90)
    print(f"SFT performance for {sft_model_path} in temperature {args.temperature}:")
    print("Average scores for {}: {}".format(sft_model_path, avg_scores))
    print(
        "Standard deviation of scores for {}: {}".format(sft_model_path, std_scores)
    )
    print("*" * 90)

    if args.eval_ppo:
        # get the relevant PPO runs using heuristics
        relevant_runs = []
        for run in os.listdir("ckpts"):
            if run.startswith(f"ppo_{args.reward}_{args.model}"):
                if run not in evaluated_runs:
                    relevant_runs.append(run)
        print(f"{len(relevant_runs)} PPO run(s) will be evaluated: {relevant_runs}")

        for run in relevant_runs:
            run_dir = os.path.join("ckpts", run)
            print(f"Starting evaluation for {run_dir}")
            for ckpt in os.listdir(run_dir):
                if ckpt.startswith("step_"):
                    ari = float(ckpt.split("_ari_")[-1])
                    if args.lower_ari_bound <= ari <= args.upper_ari_bound:
                        ckpt_path = os.path.join(run_dir, ckpt)
                        print(f"Starting evaluation for {ckpt_path}")
                        model = AutoModelForCausalLM.from_pretrained(
                            ckpt_path, torch_dtype=torch.bfloat16
                        )
                        model.to(device)

                        # evaluate with test generation config
                        eval_results = evaluate_model(
                            model,
                            dataset["test"],
                            tokenizer,
                            test_generation_config,
                            batch_size=args.batch_size,
                            model_type='clm' if args.model != 'long-t5-tglobal-xl' else 'seq2seq',
                            verbose=args.verbose
                        )
                        # save evaluation results to CSV
                        file_path = os.path.join(
                            save_dir, f"{ckpt_path.replace('/', '|')}.csv"
                        )
                        with open(file_path, mode="w", encoding="utf-8") as file:
                            writer = csv.DictWriter(file,
                                                    fieldnames=eval_results[0].keys())
                            writer.writeheader()
                            writer.writerows(eval_results)

                        # calculate average and standard deviation of scores
                        avg_scores = {
                            f"avg_{metric}": np.mean([x[metric] for x in eval_results])
                            for metric in eval_results[0].keys()
                            if metric not in ["generated_text"]
                        }
                        std_scores = {
                            f"std_{metric}": np.std([x[metric] for x in eval_results])
                            for metric in eval_results[0].keys()
                            if metric not in ["generated_text"]
                        }

                        # save the overview in JSONL format
                        with open(overview_path, mode="a", encoding="utf-8") as f:
                            json.dump(
                                {"run_path": run_dir}
                                | {"ckpt_path": ckpt_path}
                                | avg_scores
                                | std_scores,
                                f,
                            )
                            f.write("\n")

                        # print out results
                            print("*" * 90)
                            print(
                                f"RLUAM performance for {ckpt_path} in temperature "
                                f"{args.temperature}:"
                            )
                            print("Average scores for {}: {}".format(ckpt_path,
                                                                     avg_scores))
                            print(
                                "Standard deviation of scores for {}: {}".format(
                                    ckpt_path, std_scores
                                )
                            )
                            print("*" * 90)
                    print("*" * 90)
