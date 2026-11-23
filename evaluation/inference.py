import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import re
import time
import traceback

import httpx
from transformers import AutoTokenizer

from conversation import Conversation, has_inline_tools
from corpus.graph import CORPUS_FILES, DEFAULT_DATA_ROOT, GraphCorpus, Timeline
from corpus.retrieval_client import RetrievalClient
from .evaluate import evaluate
from .generation import SGLangClient


class History(Conversation):
    def __init__(self, tokenizer, question, *, system, tools):
        super().__init__(tokenizer,
            [{"role": "system", "content": system},
             {"role": "user", "content": "Question: " + question}],
            tools)
        self.seen = {}


def load_profiles(path, corpora=()):
    if path is None:
        profiles = {}
        benchmarks = {"multitq" if corpus == "MultiTQ" else "timelinekgqa" for corpus in corpora}
        for benchmark in sorted(benchmarks):
            profiles.update(load_profiles(DEFAULT_DATA_ROOT.parent / "data" / f"sft_{benchmark}.json"))
        return profiles
    profiles = {}
    for row in json.loads(path.read_text()):
        system = row["messages"][0]
        if system["role"] != "system":
            raise ValueError("SFT samples must begin with a fixed system message")
        content = system["content"]
        corpus = next((name for name, marker in [("MultiTQ", "MultiTQ"),
            ("CronQuestion", "CronQuestions"), ("ICEWS", "TimelineKGQA-ICEWS")] if marker in content), None)
        if corpus is None and not has_inline_tools(content):
            raise ValueError("Unrecognized corpus in SFT system message")
        names = {tool["function"]["name"] for tool in row["tools"]}
        role = "coordinator" if "delegate_many" in names else "worker"
        profile = dict(system=content, tools=row["tools"])
        roles = profiles.setdefault(corpus, {})
        if role in roles and roles[role] != profile:
            raise ValueError(f"Multiple fixed profiles for {corpus or 'shared TimelineKGQA'}/{role}")
        roles[role] = profile
    shared = profiles.pop(None, None)
    if shared is not None:
        timelinekgqa_corpora = ("CronQuestion", "ICEWS")
        if set(shared) != {"coordinator", "worker"} or any(c in profiles for c in timelinekgqa_corpora):
            raise ValueError("Shared TimelineKGQA data requires one fixed profile per role")
        for corpus in timelinekgqa_corpora:
            profiles[corpus] = {role: profile.copy() for role, profile in shared.items()}
    return profiles


def turn_observation(workers, turn):
    body = "\n\n---\n\n".join(f"Worker {i}\n" + message(w) for i, w in enumerate(workers, 1))
    return f"Worker messages (delegation turn {turn}):\n\n{body}"


def parse_action(text, role):
    calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    if len(calls) != 1:
        raise ValueError("Expected exactly one native tool call")
    try:
        call = json.loads(calls[0])
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid tool JSON: {error.msg}") from error
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        raise ValueError("Tool call requires name and arguments")
    name, args = call["name"], call["arguments"]
    if not isinstance(name, str) or not isinstance(args, dict):
        raise ValueError("Unknown tool or invalid arguments")
    if role == "coordinator" and name == "delegate_many":
        subtasks = args.get("subtasks")
        if (set(args) != {"subtasks"} or not isinstance(subtasks, list) or not subtasks
                or any(not isinstance(s, str) or not s.strip() for s in subtasks)):
            raise ValueError("delegate_many requires nonempty subtasks")
        return call
    required = {"answer": {"answer"}}
    if role == "worker":
        required.update(search={"query"}, timeline={"fact", "keep"})
    if name not in required:
        raise ValueError(f"Tool {name} is unavailable to {role}")
    allowed = required[name] | ({"at"} if name == "timeline" else set())
    if not required[name] <= args.keys() or set(args) - allowed:
        raise ValueError(f"Invalid {name} arguments")
    if any(not isinstance(v, str) or not v.strip() for v in args.values()):
        raise ValueError("Tool arguments must be nonempty strings")
    return call


def message(worker):
    observations = [c["tool_observation"] for c in worker["calls"] if "tool_observation" in c]
    blocks = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(observations, 1)) or "(none)"
    report = worker["answer"] if worker["status"] == "complete" else f"(no report: {worker['status']})"
    return f"Subtask: {worker['subtask']}\nTool observations:\n{blocks}\nReport:\n{report}"


def actor(history, member, subtask=None):
    return dict(history=history, member=member, subtask=subtask, calls=[],
                generated_tokens=0, status="running", answer=None)


def pack_actor(current):
    history = current["history"]
    return {**{key: value for key, value in current.items() if key != "history"},
            "input_ids": history.ids, "loss_mask": history.generated}


class Collaboration:
    def __init__(self, config, model, profiles, corpora, data_root):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.tokenizer.clean_up_tokenization_spaces = False
        self.atlases = {c: Timeline(GraphCorpus(c, data_root), window_size=config["window_size"])
                        for c in corpora}
        self.retrievals = {c: RetrievalClient(config["retrieval_url"], c, top_k=config["top_k"])
                           for c in corpora}
        self.client = SGLangClient(config["base_url"], data_parallel_size=config["data_parallel_size"])
        self.pool = ThreadPoolExecutor(max_workers=config["generation_workers"])
        self.profiles = profiles

    def history(self, role, corpus, question):
        profile = self.profiles[corpus][role]
        return History(self.tokenizer, question, system=profile["system"], tools=profile["tools"])

    def close(self):
        self.pool.shutdown(wait=True)
        self.client.close()
        for client in self.retrievals.values():
            client.close()

    async def sample(self, current, item, budget, call_limit):
        history = current["history"]
        # SGLang requires prompt + completion < context length.
        remaining = min(budget - current["generated_tokens"], self.config["context"] - 1 - len(history.ids))
        if remaining <= 0 or len(current["calls"]) >= call_limit:
            current["status"] = "budget_exhausted"
            return None
        seed = self.config["seed"] + item["sampling_index"] * 4096 + current["member"] * 64 + len(current["calls"])
        ids = list(history.ids)
        temperature = self.config["temperature"]
        if current["member"]:
            temperature = self.config.get("worker_temperature", temperature)

        def generate():
            with self.client.sticky_endpoint(session_key=f"{item['key']}:{current['member']}"):
                return self.client.generate_tokens(ids, max_tokens=remaining,
                    temperature=temperature, top_p=self.config["top_p"],
                    top_k=self.config["sampling_top_k"], min_p=self.config["min_p"],
                    seed=seed, stop=["</tool_call>"])
        reply = await asyncio.wrap_future(self.pool.submit(generate))
        if not reply.output_token_ids or len(reply.output_token_ids) > remaining:
            raise RuntimeError("Invalid native generation length")
        call = dict(prompt_tokens=len(ids), completion_tokens=len(reply.output_token_ids),
                    raw_response=history.append_sample(reply.output_token_ids), finish_reason=reply.finish_reason)
        current["calls"].append(call)
        current["generated_tokens"] += len(reply.output_token_ids)
        return call

    async def final_answer(self, coordinator, item):
        extra = self.config.get("final_answer_tokens")
        if not extra or coordinator.get("final_notice"):
            return None
        coordinator["final_notice"] = True
        coordinator["status"] = "running"
        notice = "Submit your final answer using answer."
        coordinator["history"].observe(notice, role="user")
        call = await self.sample(coordinator, item, self.config["coordinator_tokens"] + extra,
                                 self.config["delegation_turns"] + 2)
        if call is None:
            return None
        try:
            action = parse_action(call["raw_response"], "coordinator")
        except (ValueError, TypeError, AttributeError):
            return None
        if action["name"] != "answer":
            return None
        call["action"] = action
        return action

    async def investigate(self, current, item):
        cfg, history = self.config, current["history"]
        atlas = self.atlases[item["corpus"]]
        budget = current.get("token_budget", cfg["worker_tokens"])
        while current["status"] == "running":
            call = await self.sample(current, item, budget, cfg["worker_calls"])
            if call is None:
                break
            try:
                action = parse_action(call["raw_response"], "worker")
            except ValueError as error:
                current.update(status="budget_exhausted" if call["finish_reason"] == "length" else "invalid_action",
                               error=str(error))
                call["action_error"] = str(error)
                break
            call["action"] = action
            name, arguments = action["name"], action["arguments"]
            if name == "answer":
                current.update(status="complete", answer=arguments["answer"])
                break
            started = time.perf_counter()
            try:
                if name == "search":
                    result = await asyncio.to_thread(self.retrievals[item["corpus"]].access,
                        dict(name="semantic_search", parameters=arguments))
                    facts = [atlas.canonical(str(row["id"])) for row in result["facts"]]
                    history.seen.update((fact, atlas.graph.rows[fact]) for fact in facts)
                    observation = (f"Search results (relevance order): returned={len(facts)}\n"
                                   + "\n".join(atlas.fact_line(f) for f in facts))
                    call["fact_ids"] = facts
                else:
                    window = atlas.timeline(arguments["fact"], arguments["keep"], arguments.get("at"), history.seen)
                    history.seen.update((fact, atlas.graph.rows[fact]) for fact in window.ids)
                    observation = atlas.render(window)
                    call["window"] = asdict(window)
            except ValueError as error:
                observation = f"Tool error: {error}"
                call["tool_error"] = str(error)
            call.update(tool_observation=observation, tool_seconds=time.perf_counter() - started)
            history.observe(observation)

    async def episode(self, item):
        started = time.perf_counter()
        cfg, corpus = self.config, item["corpus"]
        coordinator = actor(self.history("coordinator", corpus, item["question"]), 0)
        actors, turns, status, error = [coordinator], 0, "running", None
        remaining_worker_tokens = cfg.get("shared_worker_tokens")
        try:
            while status == "running":
                call = await self.sample(coordinator, item, cfg["coordinator_tokens"], cfg["delegation_turns"] + 1)
                action = None
                if call is not None:
                    try:
                        action = parse_action(call["raw_response"], "coordinator")
                    except (ValueError, TypeError, AttributeError) as issue:
                        if call["finish_reason"] != "length" and not cfg.get("final_answer_tokens"):
                            status, error = "invalid_action", str(issue)
                            break
                        error = str(issue)
                if action is None:
                    action = await self.final_answer(coordinator, item)
                    if action is None:
                        status = "budget_exhausted"
                        break
                    error = None
                else:
                    call["action"] = action
                if action["name"] == "delegate_many" and turns == cfg["delegation_turns"]:
                    action = await self.final_answer(coordinator, item)
                    if action is None:
                        status, error = "invalid_action", "Delegation after the last delegation turn"
                        break
                if action["name"] == "answer":
                    coordinator.update(status="complete", answer=action["arguments"]["answer"])
                    status = "complete"
                    break
                workers = []
                for subtask in action["arguments"]["subtasks"]:
                    history = self.history("worker", corpus,
                        item["question"] + "\n\nYour subtask: " + subtask)
                    workers.append(actor(history, len(actors), subtask))
                    actors.append(workers[-1])
                if remaining_worker_tokens is not None:
                    share = min(cfg["worker_tokens"], remaining_worker_tokens // len(workers))
                    for worker in workers:
                        worker["token_budget"] = share
                results = await asyncio.gather(*(self.investigate(w, item) for w in workers),
                                               return_exceptions=True)
                for worker, result in zip(workers, results, strict=True):
                    if isinstance(result, BaseException):
                        worker.update(status="infrastructure_error", error=f"{type(result).__name__}: {result}")
                        raise result
                if remaining_worker_tokens is not None:
                    remaining_worker_tokens -= sum(w["generated_tokens"] for w in workers)
                turns += 1
                observation = turn_observation(workers, turns)
                call.update(workers=[w["member"] for w in workers], tool_observation=observation)
                coordinator["history"].observe(observation)
        except Exception:
            status, error = "infrastructure_error", traceback.format_exc()
        if coordinator["status"] == "running":
            coordinator["status"] = status
        agents = [pack_actor(a) for a in actors]
        return dict(question_id=item["question_id"], corpus=corpus, question=item["question"],
                    sampling_index=item["sampling_index"], status=status, answer=coordinator["answer"],
                    final_notice=bool(coordinator.get("final_notice")),
                    agents=agents, delegation_turns=turns, workers=len(agents) - 1,
                    generated_tokens=sum(a["generated_tokens"] for a in agents),
                    calls=sum(len(a["calls"]) for a in agents),
                    seconds=time.perf_counter() - started, error=error)


async def run(args):
    if len(args.corpus) != len(args.questions) or len(set(args.corpus)) != len(args.corpus):
        raise ValueError("Supply one question file per distinct corpus")
    profiles = load_profiles(args.data, args.corpus)
    if args.dp_size is None:
        async with httpx.AsyncClient(trust_env=False, timeout=30.0) as client:
            response = await client.get(args.base_url.rstrip("/").removesuffix("/v1") + "/server_info")
            response.raise_for_status()
            args.dp_size = response.json()["dp_size"]
        if type(args.dp_size) is not int or args.dp_size < 1:
            raise ValueError("Model service dp_size must be a positive integer")
    public_questions = {}
    for corpus, path in zip(args.corpus, args.questions, strict=True):
        public_questions[corpus] = [dict(corpus=corpus,
            question_id=str(row["quid" if corpus == "MultiTQ" else "id"]),
            question=row["question"], sampling_index=index * 3 + list(CORPUS_FILES).index(corpus))
            for index, row in enumerate(json.loads(path.read_text()))]
    config = dict(base_url=args.base_url, retrieval_url=args.retrieval_url,
                  data_parallel_size=args.dp_size, generation_workers=args.concurrency,
                  context=32768, coordinator_tokens=2048, worker_tokens=2048, shared_worker_tokens=2048,
                  worker_calls=6, delegation_turns=3, final_answer_tokens=512,
                  top_k=15, window_size=15, seed=42,
                  temperature=0.0, top_p=1.0, sampling_top_k=-1, min_p=0.0)
    if args.worker_temperature is not None:
        config["worker_temperature"] = args.worker_temperature
    manifest = dict(model=str(args.model.resolve()), data=str(args.data.resolve()) if args.data else None,
                    corpora=args.corpus, questions=[str(p.resolve()) for p in args.questions],
                    corpus_root=str(args.corpus_root.resolve()), public_questions=public_questions,
                    profiles={c: profiles[c] for c in args.corpus}, protocol=config)
    args.output.mkdir(parents=True, exist_ok=True)
    record = args.output / "run.json"
    if record.exists():
        if not args.resume or json.loads(record.read_text()) != manifest:
            raise ValueError("Use a new output directory or resume the identical run")
    else:
        if args.resume:
            raise ValueError("No saved run to resume")
        record.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    items, done = [], set()
    for corpus, path in zip(args.corpus, args.questions, strict=True):
        if set(profiles[corpus]) != {"coordinator", "worker"}:
            raise ValueError(f"Missing role profile for {corpus}")
        directory = args.output / corpus
        directory.mkdir(exist_ok=True)
        public = [dict(item, key=corpus + ":" + item["question_id"]) for item in public_questions[corpus]]
        if len({p["key"] for p in public}) != len(public):
            raise ValueError(f"Duplicate question IDs in {path}")
        trace_path = directory / "traces.jsonl.gz"
        if trace_path.exists():
            with gzip.open(trace_path, "rt") as stream:
                saved_ids = [json.loads(line)["question_id"] for line in stream if line.strip()]
            if len(set(saved_ids)) != len(saved_ids):
                raise ValueError("Duplicate saved trajectories")
            expected = {p["question_id"] for p in public}
            if not expected.issuperset(saved_ids):
                raise ValueError("Saved trajectory IDs differ from question file")
            done.update(corpus + ":" + qid for qid in saved_ids)
        items.extend(public)
    items.sort(key=lambda item: item["sampling_index"])
    total = len(items)
    pending = iter(item for item in items if item["key"] not in done)
    collaboration = Collaboration(config, str(args.model.resolve()), profiles, args.corpus, args.corpus_root)
    failures = []
    started = time.perf_counter()

    async def consume():
        while not failures:
            item = next(pending, None)
            if item is None:
                return
            result = await collaboration.episode(item)
            directory = args.output / item["corpus"]
            failed = result["status"] == "infrastructure_error"
            for agent in result["agents"]:
                del agent["loss_mask"]
            with gzip.open(directory / ("failures.jsonl.gz" if failed else "traces.jsonl.gz"),
                           "at", compresslevel=1) as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            if failed:
                failures.append(dict(question_id=item["question_id"], corpus=item["corpus"], error=result["error"]))
            else:
                done.add(item["key"])
                if not failures and len(done) % 100 == 0 and len(done) < total:
                    progress = dict(status="running", completed=len(done), total=total,
                                    seconds=time.perf_counter() - started)
                    (args.output / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                    print(f"{len(done)}/{total} questions completed", flush=True)

    try:
        await asyncio.gather(*(consume() for _ in range(args.concurrency)))
    finally:
        collaboration.close()
    progress = dict(status="failed" if failures else "completed", completed=len(done), total=total,
                    seconds=time.perf_counter() - started)
    if failures:
        progress["failures"] = failures
    (args.output / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    if failures:
        raise RuntimeError("Infrastructure failure saved; run was not scored")
    print(f"{len(done)}/{total} questions completed", flush=True)
    for corpus, questions in zip(args.corpus, args.questions, strict=True):
        directory = args.output / corpus
        evaluate(corpus, questions, directory / "traces.jsonl.gz", directory)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--corpus", choices=list(CORPUS_FILES), nargs="+", required=True)
    parser.add_argument("--questions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30910")
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:30920")
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dp-size", type=int)
    parser.add_argument("--concurrency", type=int, default=192)
    parser.add_argument("--worker-temperature", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.dp_size is not None and args.dp_size < 1) or args.concurrency < 1:
        parser.error("dp-size and concurrency must be positive")
    if args.worker_temperature is not None and not 0.0 <= args.worker_temperature:
        parser.error("worker-temperature must be nonnegative")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

