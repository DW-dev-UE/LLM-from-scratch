from __future__ import annotations

import json
import os

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

STATE_FILE = "trainer_state.json"


def _split_extra_state(model_sd: dict) -> tuple[dict, dict]:
    extra = {k: v for k, v in model_sd.items() if k.endswith("_extra_state")}
    return {k: v for k, v in model_sd.items() if k not in extra}, extra


def _writer(path: str) -> dcp.FileSystemWriter:
    return dcp.FileSystemWriter(path, single_file_per_rank=False)


def save_checkpoint(path: str, model, opt, trainer_state: dict) -> None:
    model_sd, optim_sd = get_state_dict(model, opt.optimizers)
    dcp.save({"model": _split_extra_state(model_sd)[0], "optim": optim_sd}, storage_writer=_writer(path))
    if dist.get_rank() == 0:
        with open(os.path.join(path, STATE_FILE), "w") as f:
            json.dump(trainer_state, f)
    dist.barrier()


def load_checkpoint(path: str, model, opt) -> dict:
    model_sd, optim_sd = get_state_dict(model, opt.optimizers)
    model_sd, extra = _split_extra_state(model_sd)
    state = {"model": model_sd, "optim": optim_sd}
    dcp.load(state, checkpoint_id=path)
    set_state_dict(model, opt.optimizers, model_state_dict={**state["model"], **extra}, optim_state_dict=state["optim"])
    with open(os.path.join(path, STATE_FILE)) as f:
        return json.load(f)


def save_offload_checkpoint(path: str, model, opt, trainer_state: dict) -> None:
    dcp.save(opt.checkpoint_state(model), storage_writer=_writer(path), process_group=getattr(opt, "process_group", None))
    if dist.get_rank() == 0:
        with open(os.path.join(path, STATE_FILE), "w") as f:
            json.dump(trainer_state, f)
    dist.barrier()


def load_offload_checkpoint(path: str, model, opt) -> dict:
    state = opt.checkpoint_state(model)
    dcp.load(state, checkpoint_id=path, process_group=getattr(opt, "process_group", None))
    opt.after_load(state)
    with open(os.path.join(path, STATE_FILE)) as f:
        return json.load(f)


def read_trainer_state(path: str) -> dict:
    p = os.path.join(path, STATE_FILE)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def load_model_weights(path: str, model) -> None:
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

    state = {"model": get_model_state_dict(model)}
    dcp.load(state, checkpoint_id=path)
    set_model_state_dict(model, state["model"])


def latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    done = [d for d in os.listdir(ckpt_dir) if os.path.exists(os.path.join(ckpt_dir, d, STATE_FILE))]
    return os.path.join(ckpt_dir, max(done)) if done else None
