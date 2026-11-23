from functools import partial
import math

import psutil
import ray
import torch
from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.metric import reduce_metrics
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding


def training_tensors(mini_batch, world_size):
    """Exclude tool observations from the training loss."""
    from tensordict import TensorDict

    rows = mini_batch["rows"]
    if not rows:
        raise ValueError("A mini-batch needs real token tapes")
    real_rows = len(rows)
    padding_rows = (-real_rows) % world_size
    # Pad ranks to equal row counts; padding contributes zero policy and KL loss.
    model_rows = list(rows)
    if padding_rows:
        source = min(
            rows, key=lambda row: len(row["input_ids"]) + len(row["output_ids"])
        )
        for _ in range(padding_rows):
            model_rows.append(
                dict(
                    input_ids=source["input_ids"],
                    output_ids=source["output_ids"],
                    response_mask=[0] * len(source["output_ids"]),
                    old_log_probs=[0.0] * len(source["output_ids"]),
                    advantage=0.0,
                )
            )

    def nested(values, dtype):
        return torch.nested.as_nested_tensor(
            [torch.tensor(v, dtype=dtype) for v in values], layout=torch.jagged
        )

    responses = [r["output_ids"] for r in model_rows]
    inputs = [r["input_ids"] + r["output_ids"] for r in model_rows]
    masks = [r["response_mask"] for r in model_rows]
    probabilities = [r["old_log_probs"] for r in model_rows]
    advantages = [
        [r["advantage"] * mask for mask in r["response_mask"]] for r in model_rows
    ]
    for index, (sequence, response, mask, old) in enumerate(
        zip(inputs, responses, masks, probabilities, strict=True)
    ):
        if (
            len(sequence) > 32768
            or len(mask) != len(response)
            or len(old) != len(response)
        ):
            raise ValueError(
                "Training tape/mask/probability shape or context capacity differs"
            )
        expected_real = index < real_rows
        if any(m not in (0, 1) for m in mask) or any(
            p != 0 for p, m in zip(old, mask) if not m
        ):
            raise ValueError(
                "Only real generated positions may carry behavior evidence"
            )
        if any(mask) != expected_real:
            raise ValueError(
                "Real rows need generated tokens and DP padding rows must be fully masked"
            )
    data = TensorDict(
        dict(
            input_ids=nested(inputs, torch.long),
            position_ids=nested([list(range(len(v))) for v in inputs], torch.long),
            prompts=nested([r["input_ids"] for r in model_rows], torch.long),
            responses=nested(responses, torch.long),
            response_mask=nested(masks, torch.float32),
            old_log_probs=nested(probabilities, torch.float32),
            advantages=nested(advantages, torch.float32),
        ),
        batch_size=[len(model_rows)],
    )
    data["loss_mask"] = data["response_mask"]
    tu.assign_non_tensor(
        data,
        request_count=mini_batch["request_count"],
        global_batch_size=real_rows,
        mini_batch_size=len(model_rows),
        epochs=1,
        seed=mini_batch["seed"],
        dataloader_kwargs={"shuffle": False},
        compute_loss=True,
        calculate_entropy=False,
        temperature=1.0,
        global_token_num=[len(v) for v in inputs],
    )
    return data


def balance_dp_tensors(data, world_size):
    """Balance ranks within a mini-batch while keeping each training record intact."""
    from verl.utils.seqlen_balancing import (
        calculate_workload,
        get_seqlen_balanced_partitions,
    )

    lengths = data["input_ids"].offsets().diff()
    partitions = get_seqlen_balanced_partitions(
        calculate_workload(lengths).tolist(), k_partitions=world_size, equal_size=True
    )
    order = [index for partition in partitions for index in partition]
    return tu.index_select_tensor_dict(data, order)


class NativePPO:
    def __init__(self, config):
        self.pool = RayResourcePool(
            process_on_nodes=[config.trainer.n_gpus_per_node],
            use_gpu=True,
            max_colocate_count=1,
            name_prefix="costa",
        )
        args = RayClassWithInitArgs(
            cls=ray.remote(ActorRolloutRefWorker),
            config=config.actor_rollout_ref,
            role="actor_rollout_ref",
        )
        self.actor = RayWorkerGroup(resource_pool=self.pool, ray_cls_with_init=args)
        self.actor.init_model()
        self.actor.set_loss_fn(
            partial(
                ppo_loss, config=omega_conf_to_dataclass(config.actor_rollout_ref.actor)
            )
        )
        self.servers = LLMServerManager.create(config=config, worker_group=self.actor)
        cfg = omega_conf_to_dataclass(
            config.actor_rollout_ref.rollout.checkpoint_engine
        )
        self.weights = CheckpointEngineManager(
            cfg, self.actor, self.servers.rollout_replicas
        )
        pids = ray.get(
            [
                worker.__ray_call__.remote(lambda _: __import__("os").getpid())
                for worker in self.servers.server_handles + self.actor.workers
            ]
        )
        self.processes = [psutil.Process(pid) for pid in pids]
        self.optimizer_steps = 0
        self.rollout_awake = True

    def sleep_rollout(self):
        if self.rollout_awake:
            self.weights.sleep_replicas()
            self.rollout_awake = False

    def start_rollout(self):
        self.sleep_rollout()
        self.weights.update_weights(global_steps=self.optimizer_steps)
        self.rollout_awake = True
        return self.servers.get_client()

    def update_once(self, mini):
        self.sleep_rollout()
        data = training_tensors(mini, self.actor.world_size)
        data = balance_dp_tensors(data, self.actor.world_size)
        tu.assign_non_tensor(data, compute_loss=False)
        reference = no_padding_2_padding(
            self.actor.compute_ref_log_prob(data)["log_probs"], data
        )
        lengths = data["responses"].offsets().diff().tolist()
        data["ref_log_prob"] = torch.nested.as_nested_tensor(
            [row[:length] for row, length in zip(reference, lengths, strict=True)],
            layout=torch.jagged,
        )
        tu.assign_non_tensor(data, compute_loss=True)
        result = self.actor.update_actor(data)
        metrics = reduce_metrics(tu.get(result, "metrics"))
        assert all(math.isfinite(metrics[key]) for key in ("grad_norm", "loss"))
        self.optimizer_steps += 1
        return dict(metrics=metrics)

    def save(self, directory):
        self.actor.save_checkpoint(
            str(directory), global_step=self.optimizer_steps, max_ckpt_to_keep=None
        )
        for kind in ("model", "optim", "extra_state"):
            for rank in range(self.actor.world_size):
                assert (
                    directory
                    / f"{kind}_world_size_{self.actor.world_size}_rank_{rank}.pt"
                ).stat().st_size > 0

    def restore(self, directory, optimizer_steps):
        self.sleep_rollout()
        self.actor.load_checkpoint(str(directory), del_local_after_load=False)
        self.optimizer_steps = optimizer_steps

    def close(self):
        for process in self.processes:
            if process.is_running():
                for child in process.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
        for worker in (
            self.servers.server_handles
            + [self.servers.global_load_balancer]
            + self.actor.workers
        ):
            ray.kill(worker, no_restart=True)
        for placement in self.pool.pgs:
            ray.util.remove_placement_group(placement)
