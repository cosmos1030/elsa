import numpy as np
import torch
from transformers import AutoTokenizer
from lib.prune import globalprune_admm
from lib.eval import eval_ppl, eval_zero_shot
from lib.utils import check_sparsity, get_llm
from lib.on_policy_distill import run_on_policy_distillation
from absl import logging, app, flags
from importlib.metadata import version
import os
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
import torch.distributed as dist
import wandb

logging.info(f"{version('torch')=}")
logging.info(f"{version('transformers')=}")
logging.info(f"{version('accelerate')=}")
logging.info(f'# of gpus: {torch.cuda.device_count()}')

FLAGS = flags.FLAGS


def main(argv):
    global FLAGS
    arguments = FLAGS.flag_values_dict()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend='nccl')

    if FLAGS.wandb and local_rank == 0:
        run_name = f"{FLAGS.model}_sparsity{FLAGS.sparsity_ratio}_admm_steps{FLAGS.admm_steps}_batch{FLAGS.admm_batch_size}_lr{FLAGS.admm_lr}"
        
        if FLAGS.do_distill:
            kl_type = "ForwardKL" if FLAGS.distill_alpha == 0.0 else ("ReverseKL" if FLAGS.distill_alpha == 1.0 else f"Alpha{FLAGS.distill_alpha}")
            run_name += f"_distill_{kl_type}_steps{FLAGS.distill_steps}"
            
        wandb.init(
            project=FLAGS.wandb_project,
            name=run_name
        )

        if not dict(wandb.config):
            wandb.config.update(arguments)
        else:
            updated_args = {
                k: wandb.config.get(k, v) for k, v in arguments.items()
            }
            FLAGS = type('FLAGS', (), updated_args)()
            logging.info(f"Updated args with wandb.config: {FLAGS}")
    else:
        if local_rank == 0:
            logging.info('\n' + '\n'.join([f'{k} = {v}' for k, v in arguments.items()]))


    # Setting seeds for reproducibility
    np.random.seed(FLAGS.seed)
    torch.random.manual_seed(FLAGS.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if FLAGS.sparsity_type != "unstructured":
        assert FLAGS.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, FLAGS.sparsity_type.split(":"))

    if local_rank == 0:
        logging.info(f"loading llm model {FLAGS.model}")

    model = get_llm(FLAGS.model, FLAGS.seqlen)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model, use_fast=False)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    model = model.to('cpu')
    model.config.use_cache = False

    logging.info(f"Process {local_rank} uses device {device}")

    if FLAGS.sparsity_ratio != 0:
        logging.info("pruning starts")
        globalprune_admm(FLAGS, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    if local_rank == 0:
        logging.info("Pruning finished")

    if is_distributed:
        dist.barrier()
        state_dict_options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        full_state = get_model_state_dict(model, options=state_dict_options)
        if local_rank == 0:
            model = get_llm(FLAGS.model, FLAGS.seqlen)
            model.load_state_dict(full_state)
        dist.destroy_process_group()

    
    if FLAGS.do_distill:
        if local_rank == 0:
            logging.info("--- Starting On-Policy Distillation Phase ---")
        # 1. 여기서 티처 모델을 직접 로드합니다. (원본 Dense 모델)
        # 메모리 효율을 위해 bfloat16을 권장하며, GPU 장치(device)로 이동시킵니다.
        teacher_model = get_llm(FLAGS.model, FLAGS.seqlen)
        teacher_model.to(torch.bfloat16).to(device)
        teacher_model.eval()

        # 2. 로드한 티처 모델을 인자로 명시적으로 넘겨줍니다.
        run_on_policy_distillation(FLAGS, model, teacher_model, tokenizer, device)

        # 3. 학습이 끝나면 티처를 메모리에서 해제하여 다음 단계(Eval 등)를 대비합니다.
        del teacher_model
        torch.cuda.empty_cache()

    if local_rank == 0:

        if "gemma-2-27b" in FLAGS.model:
            logging.info("gemma-2-27b model detected. Casting to torch.bfloat16 for stability.")
            model = model.to(torch.bfloat16)
        else:
            logging.info(f"Casting model ({FLAGS.model}) to torch.float16.")
            model = model.to(torch.float16)
        model.seqlen = FLAGS.seqlen
        model = model.to(device)
        model.eval()
        # sparsity sanity check
        logging.info("*"*30)
        sparsity_ratio = check_sparsity(model,log_by_block=True)
        logging.info(f"sparsity sanity check {sparsity_ratio:.4f}")
        logging.info("*"*30)

        # perplexity evaluation
        ppl_test = eval_ppl(FLAGS, model, tokenizer, device,data_path=FLAGS.data_path)
        logging.info([(key,ppl) for key,ppl in ppl_test.items()])
        if FLAGS.wandb:
            wandb.log({"sparsity_ratio": sparsity_ratio, **{f"ppl_test({key})": value for key, value in ppl_test.items()}})
        ## zero-shot evaluation
        if FLAGS.eval_zero_shot:
            logging.info(f"--- Evaluating After Pruning (global_admm, Zero-Shot) ---")
            accelerate = "70b" in FLAGS.model
            task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa", "piqa","race"]
            num_shot = 0
            results_after = eval_zero_shot(FLAGS, FLAGS.model, model, tokenizer, task_list, num_shot, accelerate)
            logging.info(f"Zero-shot results after pruning (global_admm):")
            logging.info(results_after)
            if FLAGS.wandb:
                for task_name, metrics in results_after.items():
                    try:
                        acc = metrics.get('acc,none', metrics.get('acc', None))
                        stderr = metrics.get('acc_stderr,none', metrics.get('acc_stderr', None))
                        if acc is not None:
                                wandb.log({f"global_admm/{task_name}_acc": acc})
                        if stderr is not None:
                                wandb.log({f"global_admm/{task_name}_stderr": stderr})
                    except Exception as log_e:
                        logging.warning(f"Could not log zero-shot metric for {task_name}: {log_e}")



if __name__ == '__main__':
    flags.DEFINE_string('model', 'facebook/opt-125m', 'model to prune. model name (hf repo) or local path to model snapshot')
    flags.DEFINE_integer('seqlen', 2048, 'Sequence length for the model.')
    flags.DEFINE_integer('seed', 0, 'Seed for sampling the calibration data.')
    flags.DEFINE_integer('nsamples', 128, 'Number of calibration samples.')
    flags.DEFINE_float('sparsity_ratio', 0.6, 'Sparsity level')
    flags.DEFINE_enum('sparsity_type', "unstructured", ["unstructured", "4:8", "2:4"], 'Type of sparsity.')
    flags.DEFINE_enum('dataset', 'c4', ["c4", "wikitext2"], 'Calibration dataset.')
    flags.DEFINE_string('data_path', None , 'Path to local snapshot (e.g., huggingface/hub/allenai-c4/snapshot/hash..)')

    # Global ADMM hyperparams
    flags.DEFINE_float('admm_beta1', 0.9, 'Beta1 for ADMM Adam optimizer.')
    flags.DEFINE_float('admm_beta2', 0.95, 'Beta2 for ADMM Adam optimizer.')
    flags.DEFINE_integer('admm_num_train_samples', 4, 'Number of training samples for ADMM.')
    flags.DEFINE_integer('admm_num_eval_samples', 4, 'Number of evaluation samples for ADMM.')
    flags.DEFINE_bool('admm_save_inputs', False , 'whether to save tokenized inputs as a cache')
    flags.DEFINE_string('admm_save_path', None, 'Path to save ADMM training results and checkpoints.')
    flags.DEFINE_bool('save_model',False, 'Whether to save the pruned model after ADMM training.')

    # Training Loop Config
    flags.DEFINE_integer('admm_epochs', 1, 'Number of epochs for ADMM training.')
    flags.DEFINE_integer('admm_steps', 10, 'Max steps for ADMM training. Overrides admm_epochs if > 0.')
    flags.DEFINE_integer('admm_batch_size', 2, 'Batch size for ADMM training, per device.')
    flags.DEFINE_integer('admm_gradient_accumulation_steps', 1, 'Gradient accumulation steps for ADMM.')
    flags.DEFINE_bool('admm_gradient_checkpointing', False, 'Use gradient checkpointing for ADMM training. Set False when using FSDP')
    flags.DEFINE_float('admm_lr', 2e-4, 'Learning rate for ADMM base optimizer.')
    flags.DEFINE_string('admm_lr_scheduler', 'linear', 'Learning rate scheduler type for ADMM.')
    flags.DEFINE_integer('admm_warmup_steps', 0, 'Warmup steps for ADMM learning rate scheduler.')
    flags.DEFINE_float('admm_weight_decay', 0.0, 'Weight decay for ADMM base optimizer.')
    flags.DEFINE_enum('admm_precision', 'bf16', ['fp32', 'fp16', 'bf16'], 'Precision for ADMM training (fp16/bf16 enables Trainer autocast).')
    flags.DEFINE_enum('admm_projection_mode', 'identity', ['identity', 'momentum'], 'objective-aware projection for ADMM.')
    flags.DEFINE_bool('admm_projection_bias_correction', False, 'Whether to use bias correction in obejctive-aware ADMM projection.')

    # ADMM Specific Config
    flags.DEFINE_float('admm_lmda', 0.01, 'Lambda penalty parameter for ADMM (for constant schedule).')
    flags.DEFINE_float('admm_init_lmda', 0.0, 'Initial lambda value for ADMM scheduling.')
    flags.DEFINE_float('admm_final_lmda', 0.01, 'Final lambda value for ADMM scheduling.')
    flags.DEFINE_bool('admm_init_lambda_from_inv_resid', False, 'Initialize lambda from inverse of initial residual.')
    flags.DEFINE_enum('admm_lmda_schedule_mode', 'constant', ['constant', 'linear', 'exponential', 'cosine'], 'Mode for lambda schedule (e.g., linear, cosine).')
    flags.DEFINE_integer('admm_interval', 2, 'Interval for ADMM projection and dual updates.')
    flags.DEFINE_enum('admm_base_optimizer', 'adam', ['adam','adamw','adam8bit','adam4bit','sgd'], 'Base optimizer for ADMM primal update.')
    flags.DEFINE_enum('admm_dual_dtype', 'fp32', ['fp32','bf16', 'float8_e4m3fn', 'float8_e5m2'], 'Dtype for ADMM dual variable (fp32 or bf16).')
    flags.DEFINE_enum('admm_split_dtype', 'fp32', ['fp32','bf16', 'float8_e4m3fn', 'float8_e5m2'], 'Dtype for ADMM split variable (fp32 or bf16).')
    flags.DEFINE_bool('admm_nonuniform_sparsity', False, 'Whether to use non-uniform sparsity based on sensitivity scores in ADMM.')
    flags.DEFINE_string('admm_nonuniform_sparsity_config_file', None, 'Path to non-uniform sparsity configuration file (JSON format).')

    # On-policy distillation
    flags.DEFINE_bool('do_distill', False, 'Whether to perform on-policy distillation after retraining.')
    flags.DEFINE_float('distill_lr', 1e-5, 'Learning rate for distillation.')
    flags.DEFINE_integer('distill_steps', 50, 'Number of distillation steps.')
    flags.DEFINE_integer('distill_batch_size', 2, 'Batch size per device for distillation.')
    flags.DEFINE_float('distill_temp', 1.0, 'Temperature for distillation.')
    flags.DEFINE_float('distill_alpha', 1.0, 'Alpha for KL (1.0 for reverse KL).')
    flags.DEFINE_integer('distill_topk', None, 'Top-k for distillation. If None, use full logits.')
    flags.DEFINE_bool('distill_add_tail', True, 'Whether to add tail for top-k distillation.')

    # Logging & Evaluation
    flags.DEFINE_integer('admm_logging_steps', 1, 'Logging step interval for ADMM training.')
    flags.DEFINE_integer('admm_eval_steps', 1, 'Evaluation step interval for ADMM training.')

    flags.DEFINE_bool('data_ablation', False, 'Whether to use data ablation, for section 5.5. If True, we fix the step size and control the number of train samples with --admm_num_train_samples.')
    flags.DEFINE_bool('eval_zero_shot', True, 'Whether to evaluate zero-shot performance.')
    flags.DEFINE_bool('wandb', False, 'Whether to use wandb for logging.')
    flags.DEFINE_string('wandb_project', None, 'wandb project name.')
    app.run(main)
