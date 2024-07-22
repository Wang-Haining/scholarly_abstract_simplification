"""
This module performs supervised finetuning on the OLMo, Gemma, and Phi-2 using the
Scientific Abstract-Significance Statement dataset (SASS). It concatenates scientific
abstracts with their simplified versions using a straightforward template.
"""

__author__ = "hw56@indiana.edu"
__version__ = "0.0.1"
__license__ = "0BSD"

import argparse
import os
from typing import List

import torch
import wandb
from datasets import DatasetDict, load_from_disk
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          EarlyStoppingCallback, TrainingArguments)
from trl import SFTTrainer, set_seed

from utils import (CKPTS_DIR, DATASET_PATH, GEMMA_2B,OLMO_1B, PHI2_3B,
                   MAX_INPUT_LENGTHS, MAX_OUTPUT_LENGTHS,
                   PROJECT_NAME, RESPONSE_TEMP, SEED, TASK_PREFIX)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def formatting_func(example: DatasetDict) -> List[str]:
    """
    Formats input examples by concatenating the source text with the target text,
    using the task-specific prefix and response template.

    Args:
        example: A dataset dictionary containing 'source' and 'target' fields.

    Returns:
        A list of formatted strings ready for model training.
    """
    output_texts = []
    for i in range(len(example["source"])):
        text = (
            TASK_PREFIX
            + f"{example['source'][i]}{RESPONSE_TEMP} {example['target'][i]}"
        )
        output_texts.append(text)

    return output_texts


if __name__ == "__main__":

    set_seed(SEED + 2122)
    parser = argparse.ArgumentParser(description="Supervise Fine-tuning with "
                                                 "Gemma-2B/7B, OLMo-1B, Llama3-8B or Phi-2.")
    parser.add_argument("--model", type=str,
                        choices=["gemma-2b", "gemma-7b", "olmo-1b", "llama3-8b", 'phi-2'],
                        help="Either gemma-2b, gemma-7b, olmo-1b, llama3-8b, gpt2-xl, or phi-2")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    args = parser.parse_args()

    if args.model == "gemma-2b":
        model_name = GEMMA_2B
    elif args.model == "olmo-1b":
        model_name = OLMO_1B
    elif args.model == "phi-2":
        model_name = PHI2_3B
    else:
        raise ValueError(f"Invalid model name: {args.model}")

    run_name = f'sft_{model_name.split("/")[-1]}'
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    dataset = load_from_disk(DATASET_PATH)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.bfloat16)
    if any(keyword in model_name.lower() for keyword in ['phi', 'llama']):
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=f"{CKPTS_DIR}/{run_name}",
        overwrite_output_dir=True,
        num_train_epochs=50.0,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,  # same to training
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        lr_scheduler_type='constant_with_warmup',
        warmup_steps=50,
        weight_decay=1e-1,
        logging_steps=20,
        eval_steps=20,
        bf16=True,
        report_to="wandb",
        load_best_model_at_end=True,
        save_steps=20,
        save_total_limit=3,
        remove_unused_columns=True,
    )
    wandb.init(project=PROJECT_NAME, name=run_name, config=training_args)

    trainer = SFTTrainer(
        model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=formatting_func,
        max_seq_length=MAX_INPUT_LENGTHS[args.model] + MAX_OUTPUT_LENGTHS[args.model] + 10,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
