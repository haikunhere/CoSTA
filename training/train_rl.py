import argparse
import asyncio
from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
TIMELINEKGQA_CORPORA = ("CronQuestion", "ICEWS")


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


class QuestionTraversal:
    def __init__(self, path, seed, corpus="MultiTQ"):
        self.corpus = corpus
        self.rows = json.loads(path.read_text())
        self.rng = random.Random(seed)
        self.order = list(range(len(self.rows)))
        self.rng.shuffle(self.order)
        self.cursor = self.epoch = 0

    def take(self, count):
        assert len(self.order) >= count
        if self.cursor + count > len(self.order):
            self.rng.shuffle(self.order)
            self.cursor = 0
            self.epoch += 1
        ids = self.order[self.cursor : self.cursor + count]
        self.cursor += count
        return ids

    def select(self, ids):
        return [(self.corpus, self.rows[i]) for i in ids]

    def state(self):
        return dict(
            order=self.order,
            cursor=self.cursor,
            epoch=self.epoch,
            rng=self.rng.getstate(),
        )

    def restore(self, state):
        assert sorted(state["order"]) == list(range(len(self.rows)))
        assert 0 <= state["cursor"] <= len(self.rows)
        self.order, self.cursor, self.epoch = (
            state["order"],
            state["cursor"],
            state["epoch"],
        )
        self.rng.setstate(state["rng"])


class TimelineKGQATraversal:
    """Interleave both training pools so each mini-batch is balanced."""

    def __init__(self, paths, seed):
        self.pools = {
            corpus: QuestionTraversal(path, seed + i, corpus)
            for i, (corpus, path) in enumerate(zip(TIMELINEKGQA_CORPORA, paths, strict=True))
        }

    def take(self, count):
        assert count % 4 == 0
        size = count // 2
        if any(pool.cursor + size > len(pool.rows) for pool in self.pools.values()):
            raise ValueError("TimelineKGQA training pools exhausted")
        selections = [pool.take(size) for pool in self.pools.values()]
        return [[corpus, index] for pair in zip(*selections, strict=True)
                for corpus, index in zip(self.pools, pair, strict=True)]

    def select(self, ids):
        return [(corpus, self.pools[corpus].rows[index]) for corpus, index in ids]

    def state(self):
        return {corpus: pool.state() for corpus, pool in self.pools.items()}

    def restore(self, state):
        for corpus, pool in self.pools.items():
            pool.restore(state[corpus])


def runner_for(cfg, model, profiles, retrieval_url):
    from transformers import AutoTokenizer
    from corpus.graph import GraphCorpus, Timeline
    from corpus.retrieval_client import RetrievalClient
    from evaluation.inference import Collaboration

    class NativeCollaboration(Collaboration):
        def __init__(self):
            self.config, self.profiles = cfg, profiles
            self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
            self.tokenizer.clean_up_tokenization_spaces = False
            self.atlases = {
                corpus: Timeline(GraphCorpus(corpus), window_size=cfg["window_size"])
                for corpus in profiles
            }
            self.retrievals = {
                corpus: RetrievalClient(retrieval_url, corpus, top_k=cfg["top_k"])
                for corpus in profiles
            }

        def start_batch(self, server, version):
            self.server, self.version = server, version
            for retrieval in self.retrievals.values():
                retrieval.reset_rotation()

        async def sample(self, current, item, budget, call_limit):
            history = current["history"]
            remaining = min(
                budget - current["generated_tokens"],
                cfg["context"] - 1 - len(history.ids),
            )
            if remaining <= 0 or len(current["calls"]) >= call_limit:
                current["status"] = "budget_exhausted"
                return None
            ids = list(history.ids)
            seed = item["rollout_seed"] + current["member"] * 64 + len(current["calls"])
            started = time.perf_counter()
            version = self.version
            result = await self.server.generate(
                request_id=f"{version}:{item['key']}:{item['rollout']}:{current['member']}",
                prompt_ids=ids,
                sampling_params=dict(
                    max_new_tokens=remaining,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=-1,
                    min_p=0.0,
                    stop=["</tool_call>"],
                    no_stop_trim=True,
                    sampling_seed=seed,
                    logprobs=True,
                ),
            )
            tokens, probs = list(result.token_ids), result.log_probs
            assert result.extra_fields["global_steps"] == version
            assert (
                0 < len(tokens) <= remaining
                and probs is not None
                and len(probs) == len(tokens)
            )
            assert all(math.isfinite(p) and p <= 0 for p in probs)
            assert result.stop_reason in ("stop", "length")
            call = dict(
                input_ids=ids,
                output_ids=tokens,
                raw_response=history.append_sample(tokens),
                finish_reason=result.stop_reason,
                rollout_log_probs=list(probs),
                weight_version=version,
                seed=seed,
                seconds=time.perf_counter() - started,
            )
            current["calls"].append(call)
            current["generated_tokens"] += len(tokens)
            return call

        def close(self):
            for retrieval in self.retrievals.values():
                retrieval.close()

    return NativeCollaboration()


def tape(agent, advantage, context, version):
    first = agent["calls"][0]["input_ids"]
    response, mask, probs = [], [], []
    for call in agent["calls"]:
        tape = first + response
        prompt, generated = call["input_ids"], call["output_ids"]
        assert call["weight_version"] == version and prompt[: len(tape)] == tape
        observation = prompt[len(tape) :]
        response.extend(observation + generated)
        mask.extend([0] * len(observation) + [1] * len(generated))
        probs.extend([0.0] * len(observation) + call["rollout_log_probs"])
    end = len(first) + len(response)
    assert first + response == agent["input_ids"][:end]
    assert [0] * len(first) + mask == agent["loss_mask"][:end] and sum(mask) == agent[
        "generated_tokens"
    ]
    # A role can exhaust its call budget immediately after a final observation.
    assert not any(agent["loss_mask"][end:])
    assert len(first) + len(response) <= context
    return dict(
        input_ids=first,
        output_ids=response,
        response_mask=mask,
        old_log_probs=probs,
        advantage=advantage,
    )


async def collect(runner, questions, cfg, batch):
    from evaluation.scoring import question_id

    episodes = []
    for i, (corpus, row) in enumerate(questions):
        qid = question_id(corpus, row)
        team_stride = 1024
        question_stride = max(4096, team_stride * cfg["teams_per_question"])
        for r in range(cfg["teams_per_question"]):
            episodes.append(
                dict(
                    key=f"{corpus}:train:{qid}",
                    question_id=qid,
                    question=row["question"],
                    corpus=corpus,
                    sampling_index=i,
                    rollout=r,
                    rollout_seed=cfg["seed"] + (batch * 1000 + i) * question_stride + r * team_stride,
                )
            )
    return await asyncio.gather(*(runner.episode(item) for item in episodes))


def allocate(questions, teams, cfg, version):
    from evaluation.scoring import question_id, score_prediction

    n = cfg["teams_per_question"]
    assert len(teams) == len(questions) * n
    grouped = []
    reports = []
    for i, (corpus, question) in enumerate(questions):
        members = teams[i * n : (i + 1) * n]
        rewards = [
            float(score_prediction(corpus, question, t)["correct"]) for t in members
        ]
        total_reward = sum(rewards)
        advantages = [r - (total_reward - r) / (n - 1) for r in rewards]
        assert math.isclose(sum(advantages), 0, abs_tol=1e-12)
        rows = []
        qid = question_id(corpus, question)
        for team, reward, advantage in zip(members, rewards, advantages, strict=True):
            assert team["corpus"] == corpus and team["question_id"] == qid
            assert team["question"] == question["question"] and not team["final_notice"]
            team["reward"], team["advantage"] = reward, advantage
            for actor in team["agents"]:
                if actor["calls"]:
                    rows.append(tape(actor, advantage, cfg["context"], version))
                else:
                    assert not actor["generated_tokens"]
        report = dict(question_id=qid, rewards=rewards, advantages=advantages)
        if corpus != "MultiTQ":
            report["corpus"] = corpus
        reports.append(report)
        grouped.append(rows)
    return grouped, reports


def write_teams(path, teams):
    with gzip.open(path, "wt", compresslevel=1) as stream:
        for team in teams:
            stream.write(json.dumps(team, ensure_ascii=False) + "\n")


def save_checkpoint(backend, directory, state, recipe, traversal):
    import numpy as np
    import torch

    directory.mkdir()
    backend.save(directory / "actor")
    torch.save(
        dict(
            recipe=recipe,
            position=state,
            traversal=traversal.state(),
            python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(),
        ),
        directory / "application.pt",
    )
    write(
        directory / "complete.json",
        dict(optimizer_steps=backend.optimizer_steps, **state),
    )


def restore_checkpoint(backend, directory, recipe, traversal):
    import numpy as np
    import torch

    complete = json.loads((directory / "complete.json").read_text())
    app = torch.load(
        directory / "application.pt", map_location="cpu", weights_only=False
    )
    assert app["recipe"] == recipe
    assert app["position"] == {
        k: v for k, v in complete.items() if k != "optimizer_steps"
    }
    assert app["position"]["mini"] == 0
    traversal.restore(app["traversal"])
    backend.restore(directory / "actor", complete["optimizer_steps"])
    random.setstate(app["python_rng"])
    np.random.set_state(app["numpy_rng"])
    torch.set_rng_state(app["torch_rng"])
    return app["position"]


def load_config(benchmark, recipe_path=None):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(config_name="rl")
    config = OmegaConf.merge(config, OmegaConf.load(CONFIG_DIR / f"rl_{benchmark.lower()}.yaml"))
    if recipe_path is not None:
        config = OmegaConf.merge(config, OmegaConf.load(recipe_path))
    return config


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--questions",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument("--benchmark", choices=("MultiTQ", "TimelineKGQA"), default="MultiTQ")
    parser.add_argument(
        "--data",
        type=Path,
    )
    parser.add_argument("--output", type=Path, default=Path("saves/rl"))
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:30920")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    timelinekgqa = args.benchmark == "TimelineKGQA"
    corpora = TIMELINEKGQA_CORPORA if timelinekgqa else ("MultiTQ",)
    if len(args.questions) != len(corpora):
        parser.error("Supply one question pool per corpus")
    if args.data is None:
        args.data = ROOT / f"data/sft_{args.benchmark.lower()}.json"
    from omegaconf import OmegaConf

    config = load_config(args.benchmark, args.recipe)
    cfg = OmegaConf.to_container(config.costa, resolve=True)
    total_batches = cfg.pop("total_batches")
    checkpoint_interval = cfg.pop("checkpoint_interval")
    if type(total_batches) is not int or total_batches < 1:
        parser.error("Recipe total_batches must be a positive integer")
    if type(checkpoint_interval) is not int or checkpoint_interval < 0:
        parser.error("Checkpoint interval must be a nonnegative integer")
    if args.resume is not None:
        completed = json.loads((args.resume / "complete.json").read_text())["batch"]
        if total_batches <= completed:
            parser.error("Recipe total_batches must exceed the saved batch")
    output = args.output.resolve()
    output.mkdir(parents=True)
    os.environ.update(
        COSTA_RL_MODEL=str(args.model.resolve()),
        COSTA_RL_OUTPUT=str(output),
        SGLANG_RETURN_ORIGINAL_LOGPROB="0",
        VERL_SELECTED_LOGITS="1",
        PYTHONPATH=str(ROOT),
    )
    import numpy as np
    import ray
    import torch
    from evaluation.inference import load_profiles
    from .rl_backend import NativePPO

    assert cfg["questions_per_batch"] == 2 * cfg["questions_per_mini"]
    for seed in (random.seed, np.random.seed, torch.manual_seed):
        seed(cfg["seed"])
    paths = [path.resolve(strict=True) for path in args.questions]
    traversal = TimelineKGQATraversal(paths, cfg["seed"]) if timelinekgqa else QuestionTraversal(paths[0], cfg["seed"])
    profiles = load_profiles(args.data)
    profiles = {corpus: profiles[corpus] for corpus in corpora}
    pool = {corpus: str(path) for corpus, path in zip(corpora, paths, strict=True)}
    stats = {corpus: path.stat() for corpus, path in zip(corpora, paths, strict=True)}
    recipe = dict(
        model=str(args.model.resolve()),
        reference=str(args.model.resolve()),
        train_questions=pool if timelinekgqa else pool["MultiTQ"],
        pool_size={corpus: s.st_size for corpus, s in stats.items()} if timelinekgqa else stats["MultiTQ"].st_size,
        pool_mtime_ns={corpus: s.st_mtime_ns for corpus, s in stats.items()} if timelinekgqa else stats["MultiTQ"].st_mtime_ns,
        sft_data=str(args.data.resolve()),
        profiles=profiles,
        retrieval_url=args.retrieval_url,
        actor_rollout_ref=OmegaConf.to_container(
            config.actor_rollout_ref, resolve=True
        ),
        gpus=config.trainer.n_gpus_per_node,
        protocol=cfg,
        final_answer_tokens=0,
    )
    (output / "config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    record = dict(status="running", phase="initializing", command=sys.argv,
                  total_batches=total_batches, checkpoint_interval=checkpoint_interval, **recipe)
    write(output / "run.json", record)
    env = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(("NCCL_", "CUDA_", "SGLANG_", "HF_", "VERL_"))
        or k
        in (
            "PYTHONPATH",
            "CXX",
            "OMP_NUM_THREADS",
            "TOKENIZERS_PARALLELISM",
            "NO_PROXY",
            "no_proxy",
        )
    }
    backend = runner = None
    try:
        ray.init(address="local", runtime_env={"env_vars": env})
        backend = NativePPO(config)
        state = dict(batch=0, mini=0)
        if args.resume is not None:
            state = restore_checkpoint(backend, args.resume.resolve(), recipe, traversal)

        runner = runner_for(cfg, args.model, profiles, args.retrieval_url)
        record["completed_batches"] = state["batch"]
        for batch in range(state["batch"] + 1, total_batches + 1):
            directory = output / f"batch_{batch:04d}"
            directory.mkdir()
            questions = traversal.select(traversal.take(cfg["questions_per_batch"]))
            version = backend.optimizer_steps
            record.update(phase="rollout", batch=batch, mini=0,
                          behavior_version=version)
            write(output / "run.json", record)
            print(f"Collecting batch {batch} from policy {version}", flush=True)
            runner.start_batch(backend.start_rollout(), version)
            teams = asyncio.run(collect(runner, questions, cfg, state["batch"]))
            try:
                if any(t["status"] == "infrastructure_error" for t in teams):
                    raise RuntimeError("Rollout infrastructure failure")
                grouped, rewards = allocate(questions, teams, cfg, version)
            except Exception:
                write_teams(directory / "failed_rollout.jsonl.gz", teams)
                raise

            batch_summary = dict(
                questions=len(questions),
                teams=len(teams),
                rollout_status=Counter(t["status"] for t in teams),
                correct_teams=sum(t["reward"] for t in teams),
                nonzero_advantage_questions=sum(any(r["advantages"]) for r in rewards),
            )
            if timelinekgqa:
                batch_summary["by_corpus"] = {}
                for corpus in corpora:
                    corpus_rewards = [r for r in rewards if r["corpus"] == corpus]
                    batch_summary["by_corpus"][corpus] = dict(
                        questions=len(corpus_rewards),
                        teams=sum(len(r["rewards"]) for r in corpus_rewards),
                        correct_teams=sum(sum(r["rewards"]) for r in corpus_rewards),
                        nonzero_advantage_questions=sum(any(r["advantages"]) for r in corpus_rewards),
                    )
            size = cfg["questions_per_mini"]
            for mini in range(2):
                start = mini * size
                rows = [row for question in grouped[start : start + size] for row in question]
                record.update(phase="update", batch=batch, mini=mini + 1,
                              behavior_version=version)
                write(output / "run.json", record)
                print(f"Updating batch {batch} mini {mini + 1}", flush=True)
                update = backend.update_once(
                    dict(rows=rows, request_count=size, seed=cfg["seed"] + state["batch"])
                )
                generated = sum(sum(r["response_mask"]) for r in rows)
                summary = dict(
                    **batch_summary,
                    optimizer_steps=backend.optimizer_steps,
                    update_questions=size,
                    update_teams=size * cfg["teams_per_question"],
                    training_rows=len(rows),
                    generated_update_tokens=generated,
                    masked_observation_tokens=sum(len(r["response_mask"]) for r in rows) - generated,
                )
                update["summary"] = summary
                write(directory / f"update_{mini + 1}.json", update)
                record["optimizer_steps"] = backend.optimizer_steps
            state = dict(batch=batch, mini=0)
            record["completed_batches"] = batch
            print(f"Completed batch {batch}/{total_batches}, optimizer steps={backend.optimizer_steps}", flush=True)
            if checkpoint_interval and batch % checkpoint_interval == 0 and batch < total_batches:
                checkpoint = output / f"checkpoint_{batch:04d}"
                record.update(phase="checkpoint", checkpoint_batch=batch)
                write(output / "run.json", record)
                save_checkpoint(backend, checkpoint, state, recipe, traversal)
                record.update(phase="checkpoint_saved", latest_checkpoint=str(checkpoint))
                write(output / "run.json", record)
        record["phase"] = "checkpoint"
        write(output / "run.json", record)
        save_checkpoint(backend, output / "checkpoint", state, recipe, traversal)
        write(output / "run.json", {**record, "status": "completed", **summary})
    except BaseException:
        write(
            output / "run.json",
            {**record, "status": "failed", "error": traceback.format_exc()},
        )
        raise
    finally:
        try:
            if runner is not None:
                runner.close()
            if backend is not None:
                backend.close()
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
