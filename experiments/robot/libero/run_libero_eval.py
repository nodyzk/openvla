# """
# run_libero_eval.py

# Runs a model in a LIBERO simulation environment.

# Usage:
#     # OpenVLA:
#     # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
#     python experiments/robot/libero/run_libero_eval.py \
#         --model_family openvla \
#         --pretrained_checkpoint <CHECKPOINT_PATH> \
#         --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
#         --center_crop [ True | False ] \
#         --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
#         --use_wandb [ True | False ] \
#         --wandb_project <PROJECT> \
#         --wandb_entity <ENTITY>
# """

# import os
# import sys
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Optional, Union

# import draccus
# import numpy as np
# import tqdm
# from libero.libero import benchmark

# import wandb

# # Append current directory so that interpreter can find experiments.robot
# sys.path.append("../..")
# from experiments.robot.libero.libero_utils import (
#     get_libero_dummy_action,
#     get_libero_env,
#     get_libero_image,
#     quat2axisangle,
#     save_rollout_video,
# )
# from experiments.robot.openvla_utils import get_processor
# from experiments.robot.robot_utils import (
#     DATE_TIME,
#     get_action,
#     get_image_resize_size,
#     get_model,
#     invert_gripper_action,
#     normalize_gripper_action,
#     set_seed_everywhere,
# )


# @dataclass
# class GenerateConfig:
#     # fmt: off

#     #################################################################################################################
#     # Model-specific parameters
#     #################################################################################################################
#     model_family: str = "openvla"                    # Model family
#     pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
#     load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
#     load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

#     center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

#     #################################################################################################################
#     # LIBERO environment-specific parameters
#     #################################################################################################################
#     task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
#     num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
#     num_trials_per_task: int = 50                    # Number of rollouts per task

#     #################################################################################################################
#     # Utils
#     #################################################################################################################
#     run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
#     local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

#     use_wandb: bool = False                          # Whether to also log results in Weights & Biases
#     wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
#     wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

#     seed: int = 7                                    # Random Seed (for reproducibility)

#     # fmt: on


# @draccus.wrap()
# def eval_libero(cfg: GenerateConfig) -> None:
#     assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
#     if "image_aug" in cfg.pretrained_checkpoint:
#         assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
#     assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

#     # Set random seed
#     set_seed_everywhere(cfg.seed)

#     # [OpenVLA] Set action un-normalization key
#     cfg.unnorm_key = cfg.task_suite_name

#     # Load model
#     model = get_model(cfg)

#     # [OpenVLA] Check that the model contains the action un-normalization key
#     if cfg.model_family == "openvla":
#         # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
#         # with the suffix "_no_noops" in the dataset name)
#         if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
#             cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
#         assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

#     # [OpenVLA] Get Hugging Face processor
#     processor = None
#     if cfg.model_family == "openvla":
#         processor = get_processor(cfg)

#     # Initialize local logging
#     run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
#     if cfg.run_id_note is not None:
#         run_id += f"--{cfg.run_id_note}"
#     os.makedirs(cfg.local_log_dir, exist_ok=True)
#     local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
#     log_file = open(local_log_filepath, "w")
#     print(f"Logging to local log file: {local_log_filepath}")

#     # Initialize Weights & Biases logging as well
#     if cfg.use_wandb:
#         wandb.init(
#             entity=cfg.wandb_entity,
#             project=cfg.wandb_project,
#             name=run_id,
#         )

#     # Initialize LIBERO task suite
#     benchmark_dict = benchmark.get_benchmark_dict()
#     task_suite = benchmark_dict[cfg.task_suite_name]()
#     num_tasks_in_suite = task_suite.n_tasks
#     print(f"Task suite: {cfg.task_suite_name}")
#     log_file.write(f"Task suite: {cfg.task_suite_name}\n")

#     # Get expected image dimensions
#     resize_size = get_image_resize_size(cfg)

#     # Start evaluation
#     total_episodes, total_successes = 0, 0
#     for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
#         # Get task
#         task = task_suite.get_task(task_id)

#         # Get default LIBERO initial states
#         initial_states = task_suite.get_task_init_states(task_id)

#         # Initialize LIBERO environment and task description
#         env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

#         # Start episodes
#         task_episodes, task_successes = 0, 0
#         for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
#             print(f"\nTask: {task_description}")
#             log_file.write(f"\nTask: {task_description}\n")

#             # Reset environment
#             env.reset()

#             # Set initial states
#             obs = env.set_init_state(initial_states[episode_idx])

#             # Setup
#             t = 0
#             replay_images = []
#             if cfg.task_suite_name == "libero_spatial":
#                 max_steps = 220  # longest training demo has 193 steps
#             elif cfg.task_suite_name == "libero_object":
#                 max_steps = 280  # longest training demo has 254 steps
#             elif cfg.task_suite_name == "libero_goal":
#                 max_steps = 300  # longest training demo has 270 steps
#             elif cfg.task_suite_name == "libero_10":
#                 max_steps = 520  # longest training demo has 505 steps
#             elif cfg.task_suite_name == "libero_90":
#                 max_steps = 400  # longest training demo has 373 steps

#             print(f"Starting episode {task_episodes+1}...")
#             log_file.write(f"Starting episode {task_episodes+1}...\n")
#             while t < max_steps + cfg.num_steps_wait:
#                 try:
#                     # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
#                     # and we need to wait for them to fall
#                     if t < cfg.num_steps_wait:
#                         obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
#                         t += 1
#                         continue

#                     # Get preprocessed image
#                     img = get_libero_image(obs, resize_size)

#                     # Save preprocessed image for replay video
#                     replay_images.append(img)

#                     # Prepare observations dict
#                     # Note: OpenVLA does not take proprio state as input
#                     observation = {
#                         "full_image": img,
#                         "state": np.concatenate(
#                             (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
#                         ),
#                     }

#                     # Query model to get action
#                     action = get_action(
#                         cfg,
#                         model,
#                         observation,
#                         task_description,
#                         processor=processor,
#                     )

#                     # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
#                     action = normalize_gripper_action(action, binarize=True)

#                     # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
#                     # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
#                     if cfg.model_family == "openvla":
#                         action = invert_gripper_action(action)

#                     # Execute action in environment
#                     obs, reward, done, info = env.step(action.tolist())
#                     if done:
#                         task_successes += 1
#                         total_successes += 1
#                         break
#                     t += 1

#                 except Exception as e:
#                     print(f"Caught exception: {e}")
#                     log_file.write(f"Caught exception: {e}\n")
#                     break

#             task_episodes += 1
#             total_episodes += 1

#             # Save a replay video of the episode
#             save_rollout_video(
#                 replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
#             )

#             # Log current results
#             print(f"Success: {done}")
#             print(f"# episodes completed so far: {total_episodes}")
#             print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
#             log_file.write(f"Success: {done}\n")
#             log_file.write(f"# episodes completed so far: {total_episodes}\n")
#             log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
#             log_file.flush()

#         # Log final results
#         print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
#         print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
#         log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
#         log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
#         log_file.flush()
#         if cfg.use_wandb:
#             wandb.log(
#                 {
#                     f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
#                     f"num_episodes/{task_description}": task_episodes,
#                 }
#             )

#     # Save local log file
#     log_file.close()

#     # Push total metrics and local log file to wandb
#     if cfg.use_wandb:
#         wandb.log(
#             {
#                 "success_rate/total": float(total_successes) / float(total_episodes),
#                 "num_episodes/total": total_episodes,
#             }
#         )
#         wandb.save(local_log_filepath)


# if __name__ == "__main__":
#     eval_libero()


"""
run_libero_eval.py  (MoE-adapter version)

Drop-in replacement for the official OpenVLA run_libero_eval.py.
Adds --moe_checkpoint, --num_experts, --bottleneck_dim flags.
When --moe_checkpoint is provided the script:
  1. Loads the frozen openvla-7b base
  2. Injects MoE adapters (matching training config)
  3. Restores adapter weights from the checkpoint
  4. Runs evaluation exactly as the official script does

When --moe_checkpoint is NOT provided the script behaves identically
to the original run_libero_eval.py (plain OpenVLA or LoRA checkpoint).

Usage (MoE):
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint openvla/openvla-7b \
        --moe_checkpoint runs/moe_spatial_sanity/moe_step030000.pt \
        --num_experts 4 \
        --bottleneck_dim 256 \
        --task_suite_name libero_spatial \
        --center_crop True \
        --use_wandb True \
        --wandb_project openvla-moe \
        --wandb_entity <ENTITY>

Usage (plain OpenVLA, unchanged):
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial \
        --center_crop True
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


os.environ["HF_HOME"]           = "/gpu-data2/nkoul/hf-cache"
os.environ["TRANSFORMERS_CACHE"] = "/gpu-data2/nkoul/hf-cache"
 
 
import draccus
import h5py
import numpy as np
import torch
import tqdm
from PIL import Image
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

# ── MoE adapter utilities ─────────────────────────────────────────────────────
from prismatic.models.backbones.llm.llama_moe import (
    inject_moe_adapters,
    freeze_base_model,
    build_inference_router_all,
    print_trainable_params,
)
from prismatic.vla.action_tokenizer import ActionTokenizer

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family:          str              = "openvla"   # Model family
    pretrained_checkpoint: Union[str, Path] = ""          # Pretrained checkpoint path
    load_in_8bit:          bool             = False        # (OpenVLA) Load with 8-bit quantization
    load_in_4bit:          bool             = False        # (OpenVLA) Load with 4-bit quantization
    center_crop:           bool             = True         # Center crop (if trained w/ random crop aug)

    # ── MoE-specific (ignored when moe_checkpoint is None) ───────────────────
    moe_checkpoint:  Optional[Path] = None   # Path to saved MoE adapter .pt file
    num_experts:     int            = 4      # Must match training
    bottleneck_dim:  int            = 256    # Must match training

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name:     str = "libero_spatial"   # libero_spatial | libero_object | libero_goal | libero_10 | libero_90
    num_steps_wait:      int = 10                 # Steps to wait for objects to stabilise
    num_trials_per_task: int = 50                 # Rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note:    Optional[str] = None                  # Extra tag for run ID
    local_log_dir:  str           = "./experiments/logs"  # Local directory for eval logs

    use_wandb:      bool = False
    wandb_project:  str  = "YOUR_WANDB_PROJECT"
    wandb_entity:   str  = "YOUR_WANDB_ENTITY"

    seed: int = 7

    # fmt: on


def _inject_moe(model, cfg: GenerateConfig) -> None:
    """
    Freeze the base model, inject MoE adapters, load checkpoint weights,
    and build the inference router. Mutates `model` in-place.

    _inference_router keys in the checkpoint are skipped here — they are
    not registered nn.Parameters so they don't exist in state_dict().
    They get correctly rebuilt by build_inference_router_all() below.
    """
    print(f"\n[MoE] Injecting adapters into base model ...")
    freeze_base_model(model)
    adapters = inject_moe_adapters(
        model,
        num_experts=cfg.num_experts,
        bottleneck_dim=cfg.bottleneck_dim,
        num_tasks=1,   # single router — matches sanity-check training
    )

    # Load checkpoint manually — skip _inference_router keys
    print(f"[MoE] Loading adapter weights from: {cfg.moe_checkpoint}")
    state     = torch.load(str(cfg.moe_checkpoint), map_location="cpu", weights_only=False)
    cur_state = model.state_dict()
    matched   = 0
    for k, v in state.items():
        if "_inference_router" in k:
            continue   # not a state_dict param — rebuilt below
        if k in cur_state:
            cur_state[k].copy_(v)
            matched += 1
    eligible = sum(1 for k in state if "_inference_router" not in k)
    print(f"[MoE] Loaded {matched}/{eligible} tensors")

    # Move adapters to GPU in bfloat16 to match frozen base model
    for adapter in adapters:
        adapter.to(device=DEVICE, dtype=torch.bfloat16)

    # Build merged inference router on the correct device
    build_inference_router_all(adapters, strategy="mean")
    print_trainable_params(model)
    print(f"[MoE] Model ready for inference.\n")


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model (same as official script)
    model = get_model(cfg)

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # ── MoE: inject adapters on top of the loaded base model ─────────────────
    if cfg.moe_checkpoint is not None:
        _inject_moe(model, cfg)
        # MoE uses its own Q1/Q99 normalisation — load stats from HDF5
        _data_dir = Path("/gpu-data2/nkoul/LIBERO/datasets/libero_spatial")
        _all_actions = []
        for _hdf5_path in sorted(_data_dir.glob("*.hdf5")):
            with h5py.File(str(_hdf5_path), "r") as _f:
                for _demo_key in sorted(_f["data"].keys()):
                    _all_actions.append(_f["data"][_demo_key]["actions"][()])
        _all_np   = np.concatenate(_all_actions, axis=0)
        _q01      = np.percentile(_all_np[:, :6], 1,  axis=0)
        _q99      = np.percentile(_all_np[:, :6], 99, axis=0)
        norm_min  = np.append(_q01, -1.0).astype(np.float32)
        norm_max  = np.append(_q99,  1.0).astype(np.float32)
        action_tokenizer = ActionTokenizer(processor.tokenizer)
        print(f"[MoE] Norm stats  min={np.round(norm_min, 3)}  max={np.round(norm_max, 3)}")
    else:
        # [OpenVLA] Check that the model contains the action un-normalization key
        if cfg.model_family == "openvla":
            if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
                cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
            assert cfg.unnorm_key in model.norm_stats, (
                f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"
            )

    # Initialize local logging
    moe_tag   = f"-MoE-{Path(cfg.moe_checkpoint).stem}" if cfg.moe_checkpoint else ""
    run_id    = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}{moe_tag}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict    = benchmark.get_benchmark_dict()
    task_suite        = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220   # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280   # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300   # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520   # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400   # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator
                    # drops objects and we need to wait for them to stabilise
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    observation = {
                        "full_image": img,
                        "state": np.concatenate((
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )),
                    }

                    # Query model to get action
                    if cfg.moe_checkpoint is not None:
                        # MoE path: manual forward pass + custom Q1/Q99 unnorm
                        inputs = processor(
                            task_description, Image.fromarray(img)
                        ).to(DEVICE, dtype=torch.bfloat16)
                        with torch.inference_mode():
                            generated_ids = model.generate(
                                **inputs, max_new_tokens=7, do_sample=False
                            )
                        action_token_ids = generated_ids[0, -7:].cpu().numpy()
                        action_norm = action_tokenizer.decode_token_ids_to_actions(
                            action_token_ids
                        )
                        # Inverse of Q1/Q99 normalisation used during training
                        action = (
                            (action_norm + 1.0) / 2.0
                            * (norm_max - norm_min + 1e-8)
                            + norm_min
                        )
                    else:
                        action = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                        )

                    # Normalize gripper action [0,1] -> [-1,+1]
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] Flip gripper sign to match environment convention
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save replay video
            save_rollout_video(
                replay_images, total_episodes,
                success=done, task_description=task_description,
                log_file=log_file,
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log per-task results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log({
                f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                f"num_episodes/{task_description}": task_episodes,
            })

    # Save local log file
    log_file.close()

    # Push total metrics to wandb
    if cfg.use_wandb:
        wandb.log({
            "success_rate/total": float(total_successes) / float(total_episodes),
            "num_episodes/total": total_episodes,
        })
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
