"""
MoE-Adapter evaluation on LIBERO-Spatial.

Mirrors the structure of OpenVLA's experiments/robot/libero/run_libero_eval.py,
but loads the frozen OpenVLA base + your MoE adapter checkpoint instead of a
LoRA/full-finetune checkpoint.

Usage
-----
python eval_moe_libero_spatial.py \
    --vla_path          openvla/openvla-7b \
    --moe_checkpoint    runs/moe_spatial_sanity/moe_step030000.pt \
    --data_root_dir     /gpu-data2/nkoul/LIBERO/datasets/libero_spatial \
    --task_suite_name   libero_spatial \
    --num_trials_per_task 50 \
    --center_crop       True \
    --num_experts       4 \
    --bottleneck_dim    256 \
    --seed              7

The script will print per-task success rates and an aggregate over all 10 tasks,
matching the reporting convention in the OpenVLA paper (Table in Appendix E).

W&B logging:
    --use_wandb True --wandb_project openvla-moe --wandb_entity <your-entity>
"""
import os
import sys
# ── keep HF from writing to a shared read-only cache ──────────────────────────
os.environ["HF_HOME"]            = "/gpu-data2/nkoul/hf-cache"
os.environ["TRANSFORMERS_CACHE"] = "/gpu-data2/nkoul/hf-cache"
# ── make LIBERO importable regardless of editable-install issues ───────────────
sys.path.insert(0, "/gpu-data2/nkoul/LIBERO")

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

import draccus

# ── OpenVLA internals ─────────────────────────────────────────────────────────
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.llama_moe import (
    inject_moe_adapters,
    freeze_base_model,
    load_moe_checkpoint,
    build_inference_router_all,
    print_trainable_params,
)
from prismatic.vla.action_tokenizer import ActionTokenizer

# ── LIBERO ────────────────────────────────────────────────────────────────────
try:
    import libero.libero.envs  # noqa: F401  -- register LIBERO environments
    from libero.libero import benchmark as libero_benchmark
    from libero.libero.envs import OffScreenRenderEnv
except ImportError as e:
    raise ImportError(
        "LIBERO not found. Install it:\n"
        "  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git\n"
        "  cd LIBERO && pip install -e .\n"
        "and make sure `libero` is importable."
    ) from e


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    # ── Model ──────────────────────────────────────────────────────────────────
    vla_path:       str  = "openvla/openvla-7b"
    moe_checkpoint: Path = Path("runs/moe_spatial_sanity/moe_step030000.pt")

    # ── MoE adapter hyper-params  (must match training) ───────────────────────
    num_experts:    int  = 4
    bottleneck_dim: int  = 256

    # ── Evaluation ────────────────────────────────────────────────────────────
    task_suite_name:    str  = "libero_spatial"
    num_trials_per_task: int = 50
    max_steps_per_trial: int = 600        # LIBERO-Spatial default horizon
    center_crop:        bool = True       # match training augmentation
    seed:               int  = 7

    # ── Action un-normalization ────────────────────────────────────────────────
    # Point at the same raw HDF5 directory used during training so we can
    # re-compute the Q1/Q99 bounds.  If you saved norm_min/norm_max to a JSON
    # alongside the checkpoint, set norm_stats_path instead.
    data_root_dir:   Optional[Path] = Path("/gpu-data2/nkoul/LIBERO/datasets/libero_spatial")
    norm_stats_path: Optional[Path] = None   # e.g. runs/moe_spatial_sanity/norm_stats.json

    # ── Logging ───────────────────────────────────────────────────────────────
    use_wandb:     bool = False
    wandb_project: str  = "openvla-moe"
    wandb_entity:  str  = "your-entity"

    # ── Misc ──────────────────────────────────────────────────────────────────
    save_results_path: Optional[Path] = None  # JSON path; auto-derived if None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _center_crop_image(img: np.ndarray, crop_scale: float = 0.9) -> np.ndarray:
    """Take the center `crop_scale` × crop of an HWC uint8 image."""
    h, w = img.shape[:2]
    ch, cw = int(h * crop_scale), int(w * crop_scale)
    top  = (h - ch) // 2
    left = (w - cw) // 2
    return img[top: top + ch, left: left + cw]


def _load_norm_stats_from_hdf5(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Re-derive Q1/Q99 normalisation bounds from the raw HDF5 files.
    Identical to the logic in LiberoHDF5Dataset.__init__."""
    import h5py

    all_actions = []
    for hdf5_path in sorted(Path(data_dir).glob("*.hdf5")):
        with h5py.File(str(hdf5_path), "r") as f:
            for demo_key in sorted(f["data"].keys()):
                all_actions.append(f["data"][demo_key]["actions"][()])

    all_np = np.concatenate(all_actions, axis=0)
    q01 = np.percentile(all_np[:, :6], 1,  axis=0)
    q99 = np.percentile(all_np[:, :6], 99, axis=0)
    norm_min = np.append(q01, -1.0).astype(np.float32)
    norm_max = np.append(q99,  1.0).astype(np.float32)
    print(f"[Eval] Norm stats  min={np.round(norm_min, 4)}  max={np.round(norm_max, 4)}")
    return norm_min, norm_max


def _load_norm_stats_from_json(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        d = json.load(f)
    return np.array(d["norm_min"], dtype=np.float32), np.array(d["norm_max"], dtype=np.float32)


def _unnorm_action(action_norm: np.ndarray,
                   norm_min: np.ndarray,
                   norm_max: np.ndarray) -> np.ndarray:
    """Inverse of the normalisation applied in LiberoHDF5Dataset._normalize."""
    return (action_norm + 1.0) / 2.0 * (norm_max - norm_min + 1e-8) + norm_min


def _get_libero_task_list(suite_name: str) -> list:
    """Return the ordered list of LIBERO Task objects for a suite."""
    bm = libero_benchmark.get_benchmark_dict()[suite_name]()
    return [bm.get_task(i) for i in range(bm.get_num_tasks())]


def _make_env(task, resolution: int = 256) -> OffScreenRenderEnv:
    """Instantiate a LIBERO off-screen environment for `task`."""
    import libero.libero
    bddl_root = Path(libero.libero.__file__).parent / "bddl_files"
    bddl_path = bddl_root / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(bddl_path),
        "camera_heights":  resolution,
        "camera_widths":   resolution,
    }
    return OffScreenRenderEnv(**env_args)


def _obs_to_image(obs: dict) -> np.ndarray:
    """Extract agentview RGB from a LIBERO obs dict (HWC uint8, already right-side up)."""
    # LIBERO returns images with key 'agentview_image' (or similar)
    for key in ("agentview_image", "agentview_rgb", "image"):
        if key in obs:
            img = obs[key]
            # LIBERO sometimes renders upside-down (same as the HDF5 data)
            return img[::-1, ::-1].copy()
    raise KeyError(f"Cannot find image key in obs. Keys: {list(obs.keys())}")


# ── Inference ─────────────────────────────────────────────────────────────────

def get_action(
    vla,
    processor,
    action_tokenizer: ActionTokenizer,
    image: np.ndarray,
    task_description: str,
    norm_min: np.ndarray,
    norm_max: np.ndarray,
    center_crop: bool,
    device: torch.device,
) -> np.ndarray:
    """Run one forward pass; return an unnormalised 7-DoF action."""

    if center_crop:
        image = _center_crop_image(image, crop_scale=0.9)

    pil_img = Image.fromarray(image)

    # Format prompt exactly as in training
    prompt = f"In: What action should the robot take to {task_description}?\nOut:"

    inputs = processor(prompt, pil_img).to(device, dtype=torch.bfloat16)

    # Greedy decode — bypass predict_action (requires unnorm_key) and decode manually
    with torch.inference_mode():
        generated_ids = vla.generate(
            **inputs,
            max_new_tokens=7,   # 7-DoF action
            do_sample=False,
        )

    # Extract only the newly generated token IDs (strip the prompt)
    action_token_ids = generated_ids[0, -7:].cpu().numpy()
    action_norm = action_tokenizer.decode_token_ids_to_actions(action_token_ids)  # shape (7,) in [-1, 1]

    return _unnorm_action(action_norm, norm_min, norm_max)


# ── Evaluation loop ───────────────────────────────────────────────────────────

@draccus.wrap()
def eval_moe(cfg: EvalConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb_run = None
    if cfg.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=f"eval-moe-{Path(cfg.moe_checkpoint).stem}",
            config=vars(cfg),
        )

    # ── Register HF classes ───────────────────────────────────────────────────
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # ── Load processor ────────────────────────────────────────────────────────
    print(f"[Eval] Loading processor from {cfg.vla_path} ...")
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # ── Load base VLA (frozen) ────────────────────────────────────────────────
    print(f"[Eval] Loading base VLA from {cfg.vla_path} ...")
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # ── Inject & restore MoE adapters ────────────────────────────────────────
    freeze_base_model(vla)
    adapters = inject_moe_adapters(
        vla,
        num_experts=cfg.num_experts,
        bottleneck_dim=cfg.bottleneck_dim,
        num_tasks=1,   # single router, matching training
    )

    print(f"[Eval] Loading MoE checkpoint: {cfg.moe_checkpoint}")
    load_moe_checkpoint(vla, str(cfg.moe_checkpoint))

    # Cast adapters to bfloat16 to match the frozen base model dtype
    for adapter in adapters:
        adapter.to(torch.bfloat16)

    # Build the merged inference router (mean of all task routers — with one
    # router this is just a no-op copy, but keeps the code path consistent)
    build_inference_router_all(adapters, strategy="mean")
    print_trainable_params(vla)

    vla = vla.to(device)
    vla.eval()

    # ── Normalisation stats ───────────────────────────────────────────────────
    if cfg.norm_stats_path is not None:
        norm_min, norm_max = _load_norm_stats_from_json(cfg.norm_stats_path)
    elif cfg.data_root_dir is not None:
        norm_min, norm_max = _load_norm_stats_from_hdf5(cfg.data_root_dir)
    else:
        raise ValueError(
            "Must provide either --data_root_dir (for HDF5 stats) "
            "or --norm_stats_path (for a pre-saved JSON)."
        )

    # ── LIBERO task suite ─────────────────────────────────────────────────────
    print(f"[Eval] Loading task suite: {cfg.task_suite_name}")
    tasks = _get_libero_task_list(cfg.task_suite_name)
    print(f"[Eval] {len(tasks)} tasks found.")

    # ── Run evaluation ────────────────────────────────────────────────────────
    all_results: dict[str, dict] = {}  # task_name → {successes, trials, rate}
    total_successes = 0
    total_trials    = 0

    for task_idx, task in enumerate(tasks):
        task_name = task.name
        print(f"\n{'='*60}")
        print(f"[Eval] Task {task_idx+1}/{len(tasks)}: {task_name}")
        print(f"{'='*60}")

        env = _make_env(task)
        successes = 0

        for trial in range(cfg.num_trials_per_task):
            obs = env.reset()
            # LIBERO env returns a dict; extract state and set init state for
            # reproducible evaluation across different trial indices
            env.seed(cfg.seed + task_idx * 1000 + trial)

            done = False
            for _step in range(cfg.max_steps_per_trial):
                image = _obs_to_image(obs)
                action = get_action(
                    vla=vla,
                    processor=processor,
                    action_tokenizer=action_tokenizer,
                    image=image,
                    task_description=task_name.replace("_", " ").lower(),
                    norm_min=norm_min,
                    norm_max=norm_max,
                    center_crop=cfg.center_crop,
                    device=device,
                )

                obs, reward, done, info = env.step(action)

                if done:
                    break

            success = bool(done and reward > 0)
            # Some LIBERO envs signal success via info dict
            if not success and isinstance(info, dict):
                success = bool(info.get("success", False))

            successes += int(success)
            if (trial + 1) % 10 == 0 or (trial + 1) == cfg.num_trials_per_task:
                print(
                    f"  Trial {trial+1:3d}/{cfg.num_trials_per_task}  "
                    f"successes so far: {successes}"
                )

        env.close()

        success_rate = successes / cfg.num_trials_per_task
        all_results[task_name] = {
            "successes": successes,
            "trials":    cfg.num_trials_per_task,
            "rate":      success_rate,
        }
        total_successes += successes
        total_trials    += cfg.num_trials_per_task

        print(
            f"[Eval] Task {task_idx+1} success rate: "
            f"{successes}/{cfg.num_trials_per_task} = {success_rate*100:.1f}%"
        )

        if wandb_run is not None:
            wandb_run.log({f"eval/{task_name}/success_rate": success_rate})

    # ── Aggregate ─────────────────────────────────────────────────────────────
    aggregate_rate = total_successes / total_trials if total_trials > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"[Eval] ===  LIBERO-Spatial  |  MoE Checkpoint: {Path(cfg.moe_checkpoint).name}  ===")
    print(f"{'='*60}")
    for task_name, r in all_results.items():
        print(f"  {task_name:<55s}  {r['rate']*100:5.1f}%  ({r['successes']}/{r['trials']})")
    print(f"{'='*60}")
    print(f"  {'AGGREGATE':<55s}  {aggregate_rate*100:5.1f}%  ({total_successes}/{total_trials})")
    print(f"{'='*60}\n")

    if wandb_run is not None:
        wandb_run.log({"eval/aggregate_success_rate": aggregate_rate})

    # ── Save JSON results ─────────────────────────────────────────────────────
    results_path = cfg.save_results_path
    if results_path is None:
        ckpt_stem    = Path(cfg.moe_checkpoint).stem
        results_path = Path(cfg.moe_checkpoint).parent / f"eval_{ckpt_stem}.json"

    results_payload = {
        "checkpoint":      str(cfg.moe_checkpoint),
        "task_suite":      cfg.task_suite_name,
        "num_trials":      cfg.num_trials_per_task,
        "seed":            cfg.seed,
        "aggregate_rate":  aggregate_rate,
        "per_task":        all_results,
    }
    with open(results_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"[Eval] Results saved to: {results_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    eval_moe()