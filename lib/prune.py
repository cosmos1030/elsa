import time
import torch
import math
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import get_loaders, get_dataset
from .utils import *
from .trainer import ADMMTrainer, Trainer
from absl import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from transformers import TrainingArguments, Trainer
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

### Global pruning

# --- Define AdmmTrainingArguments ---
@dataclass
class AdmmTrainingArguments(TrainingArguments):
    """
    Training arguments specific for ADMM training, inheriting from standard TrainingArguments.
    """
    wandb: bool = field(default=False, metadata={"help": "Use wandb for logging."})
    admm_lmda: float = field(default=0.001, metadata={"help": "Lambda (rho) penalty parameter for ADMM (constant schedule)."})
    admm_init_lmda: float = field(default=0.0, metadata={"help": "Initial lambda for ADMM scheduling."})
    admm_final_lmda: float = field(default=0.01, metadata={"help": "Final lambda for ADMM scheduling."})
    admm_init_lambda_from_inv_resid: bool = field(default=False, metadata={"help": "Initialize lambda from inverse of initial residual."})
    admm_lmda_schedule_mode: str = field(default='constant', metadata={"help": "Mode for lambda schedule (linear/cosine/exponential/constant)."})
    admm_interval: int = field(default=32, metadata={"help": "Interval for ADMM projection and dual updates."})
    admm_projection_mode: str = field(default='identity', metadata={"help": "Projection mode for ADMM (identity/momentum)."})
    admm_projection_bias_correction: bool = field(default=False, metadata={"help": "Use bias correction in ADMM projection (for momentum)."})
    prune_n: int = field(default=0, metadata={"help": "N for N:M sparsity."})
    prune_m: int = field(default=0, metadata={"help": "M for N:M sparsity."})
    sparsity_ratio: float = field(default=0.0, metadata={"help": "Target sparsity ratio (for reference)."})
    admm_nonuniform_sparsity: bool = field(default=False, metadata={"help": "Use non-uniform sparsity(based on sensitivity score) in ADMM."})
    nonuniform_sparsity_config_file: str = field(default='', metadata={"help": "Config file for non-uniform sparsity in ADMM (JSON format)."})
    base_optimizer_type: str = field(default='adam', metadata={"help": "Base optimizer for ADMM primal update."})
    admm_dual_dtype: str = field(default='fp32', metadata={"help": "Dtype for ADMM dual variable (fp32 or bf16)."})
    admm_split_dtype: str = field(default='fp32', metadata={"help": "Dtype for ADMM split variable (fp32 or bf16)."})
    admm_beta1: float = field(default=0.9, metadata={"help": "Beta1 for ADMM Adam optimizer."})
    admm_beta2: float = field(default=0.95, metadata={"help": "Beta2 for ADMM Adam optimizer."})

# --- globalprune_admm function ---
def globalprune_admm(FLAGS, model, tokenizer, device, prune_n=0, prune_m=0):
    """
    Performs ADMM training globally.
    """
    if FLAGS.admm_save_path:
        model_name_part = FLAGS.model.split('/')[-1]
        admm_run_name = f"{model_name_part}_pruned{FLAGS.sparsity_ratio}_admm_lr{FLAGS.admm_lr}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        admm_output_dir = Path(FLAGS.admm_save_path) / admm_run_name
        admm_output_dir.mkdir(parents=True, exist_ok=True)
        admm_output_dir_str = str(admm_output_dir)
    else:
        admm_output_dir_str = f"./admm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    admm_training_args = AdmmTrainingArguments(
        wandb=FLAGS.wandb,
        output_dir=admm_output_dir_str,
        num_train_epochs=FLAGS.admm_epochs,
        max_steps=FLAGS.admm_steps if FLAGS.admm_steps > 0 else -1,
        per_device_train_batch_size=FLAGS.admm_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=FLAGS.admm_gradient_accumulation_steps,
        learning_rate=FLAGS.admm_lr,
        lr_scheduler_type=FLAGS.admm_lr_scheduler,
        warmup_steps=FLAGS.admm_warmup_steps,
        weight_decay=FLAGS.admm_weight_decay,
        gradient_checkpointing=FLAGS.admm_gradient_checkpointing,
        fp16=(FLAGS.admm_precision == 'fp16'),
        bf16=(FLAGS.admm_precision == 'bf16' and torch.cuda.is_bf16_supported()),
        logging_steps=FLAGS.admm_logging_steps,
        eval_strategy= "steps",
        logging_strategy="steps",
        eval_steps=FLAGS.admm_eval_steps,
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="eval_ce_loss",
        greater_is_better=False,
        report_to="wandb" if has_wandb and FLAGS.wandb else "none",
        remove_unused_columns=False,
        do_train=True,
        do_eval=True,
        # admm arguments
        admm_nonuniform_sparsity=FLAGS.admm_nonuniform_sparsity,
        nonuniform_sparsity_config_file=FLAGS.admm_nonuniform_sparsity_config_file,
        admm_lmda=FLAGS.admm_lmda,
        admm_init_lmda=FLAGS.admm_init_lmda,
        admm_final_lmda=FLAGS.admm_final_lmda,
        admm_init_lambda_from_inv_resid=FLAGS.admm_init_lambda_from_inv_resid,
        admm_lmda_schedule_mode=FLAGS.admm_lmda_schedule_mode,
        sparsity_ratio=FLAGS.sparsity_ratio,
        admm_interval=FLAGS.admm_interval,
        base_optimizer_type=FLAGS.admm_base_optimizer,
        ## admm projection
        admm_projection_mode=FLAGS.admm_projection_mode,
        admm_projection_bias_correction=FLAGS.admm_projection_bias_correction,
        prune_n=prune_n,
        prune_m=prune_m,
        admm_dual_dtype=FLAGS.admm_dual_dtype,
        admm_split_dtype=FLAGS.admm_split_dtype,
        admm_beta1=FLAGS.admm_beta1,
        admm_beta2=FLAGS.admm_beta2,
    )

    # --- Log on main process only ---
    if admm_training_args.local_rank == 0:
        logging.info("--- Starting Global Pruning via ADMM Training  ---")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        if admm_training_args.local_rank == 0:
            logging.info(f"Tokenizer pad_token not set. Using eos_token as pad_token.")

    # 1. Prepare Datasets for ADMM Training
    if admm_training_args.local_rank == 0:
        logging.info("Preparing dataset for ADMM training...")
    # Use ADMM specific flags for dataset parameters
    if FLAGS.admm_steps > 0:
        num_train_samples = FLAGS.admm_steps * FLAGS.admm_batch_size * FLAGS.admm_gradient_accumulation_steps * admm_training_args.world_size
    else:
        num_train_samples = FLAGS.admm_num_train_samples
    if FLAGS.data_ablation:
        num_train_samples = FLAGS.admm_num_train_samples
    # Ensure model's seqlen matches the one used for dataset processing
    model.seqlen = FLAGS.seqlen
    train_inputs = get_dataset(
        dataset_name=FLAGS.dataset,
        tokenizer=tokenizer,
        nsamples=num_train_samples,
        seed=FLAGS.seed,
        seqlen=model.seqlen,
        data_type="train",
        save_to_cache=FLAGS.admm_save_inputs,
        data_path=FLAGS.data_path
    )
    valid_inputs = get_dataset(
        dataset_name=FLAGS.dataset,
        tokenizer=tokenizer,
        nsamples=FLAGS.admm_num_eval_samples,
        seed=FLAGS.seed,
        seqlen=model.seqlen,
        data_type="validation",
        save_to_cache=FLAGS.admm_save_inputs,
        data_path=FLAGS.data_path
    )

    if admm_training_args.local_rank == 0:
        logging.info(f"ADMM Datasets prepared: Train size {len(train_inputs)}, Valid size {len(valid_inputs)}")

    model.train()
    # 4. Initialize ADMMTrainer
    if admm_training_args.local_rank == 0:
        logging.info("Initializing ADMMTrainer...")
    trainer = ADMMTrainer(
        model=model, # Pass the model while it's on the CPU
        args=admm_training_args,
        train_dataset=train_inputs,
        eval_dataset=valid_inputs,
        tokenizer=tokenizer,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
    )

    # 5. Start ADMM Training
    if admm_training_args.local_rank == 0:
        logging.info("Starting ADMM training on all processes...")
    trainer.train()

    if FLAGS.save_model:
        trainer.save_model(admm_output_dir_str)
        logging.info(f"ADMM pruned model saved to {admm_output_dir_str}")
