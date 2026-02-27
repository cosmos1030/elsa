
import torch
from transformers import TrainingArguments, Trainer
from dataclasses import dataclass, field
from .optimizers import MaskedAdam
from .data import get_dataset
import os
import torch.distributed as dist
from transformers.optimization import get_scheduler
from transformers.utils import is_sagemaker_mp_enabled
import math
from .trainer import OnPolicyDistillTrainer
from argparse import Namespace
from typing import Any, Optional, Dict, Tuple # 타입 힌트용

@dataclass
class OnPolicyDistillTrainingArguments(TrainingArguments):
    on_policy_distill_learning_rate: float = field(default=2e-5, metadata={"help": "Learning rate for the MaskedAdam optimizer."})
    on_policy_distill_batch_size: int = field(default=2, metadata={"help": "The batch size per device for retraining."})
    on_policy_distill_steps: int = field(default=100, metadata={"help": "The number of training steps for retraining."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Number of gradient accumulation steps."})
    distill_temp: float = field(default=1.0, metadata={"help": "Temperature for distillation."})


def run_on_policy_distillation(FLAGS, model, teacher_model, tokenizer, device):
    """
    외부에서 생성된 teacher_model을 받아 Trainer를 실행합니다.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # TrainingArguments 설정
    distill_args = OnPolicyDistillTrainingArguments(
        output_dir="./distill_output",
        learning_rate=FLAGS.distill_lr,
        per_device_train_batch_size=FLAGS.distill_batch_size,
        max_steps=FLAGS.distill_steps,
        distill_temp=FLAGS.distill_temp,
        bf16=True,
        gradient_accumulation_steps=FLAGS.admm_gradient_accumulation_steps,
        logging_steps=10,
        report_to="wandb" if FLAGS.wandb else [],
        save_strategy="no"
    )

    # 데이터셋 준비 (앞서 설명한 것과 동일)
    num_samples = distill_args.max_steps * distill_args.per_device_train_batch_size * distill_args.gradient_accumulation_steps * world_size
    train_dataset = get_dataset(
        FLAGS.dataset, tokenizer, nsamples=num_samples, 
        seed=FLAGS.seed, seqlen=FLAGS.seqlen, data_path=FLAGS.data_path
    )

    # Distill Config (Loss 계산용)
    distill_cfg = Namespace(
        full_logit_distillation=True,
        distillation_topk=FLAGS.distill_topk,
        distillation_add_tail=FLAGS.distill_add_tail,
        alpha=FLAGS.distill_alpha,
        is_clip=None
    )

    # Trainer 초기화 (이미 로드된 teacher_model 사용)
    trainer = OnPolicyDistillTrainer(
        model=model,                 # Student (Pruned)
        teacher_model=teacher_model, # Teacher (Dense)
        distill_config=distill_cfg,
        tokenizer=tokenizer,
        args=distill_args,
        train_dataset=train_dataset,
    )

    trainer.train()