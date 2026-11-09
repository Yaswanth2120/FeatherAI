"""Quantization study v0.2: source -> Feather conversions -> validation -> benchmark -> quality -> PPL/KLD.

Resumable: each phase checkpoints into <out>/study.json and is skipped when already complete.
Phases never overwrite previous artifacts (exclusive directories).
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time

from backend.core.artifacts import inspect_gguf, sha256
from backend.core.hardware import inspect as hardware
from backend.diagnostics.memory import system_snapshot
from backend.evaluation.runner import compare_dataset_evaluations, evaluate_dataset
from backend.quantization.gguf_adapter import convert
from backend.study import benchmark, perplexity
from backend.study.server import ServerSession, docker_vm_running

PHASES = ("convert", "validate", "dryrun", "benchmark", "quality", "perplexity", "report")
VALIDATION_PROMPT = "What is 2 + 2? Answer with only the number."
TARGETS = ("Q8_0", "Q4_K_M")


def _git(root: Path, *args) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).stdout.strip()


def _tree_digest(root: Path) -> str:
    """SHA-256 over `git diff HEAD` plus the names and contents of untracked files."""
    import hashlib
    digest = hashlib.sha256(subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=root, capture_output=True).stdout)
    for name in sorted(_git(root, "ls-files", "--others", "--exclude-standard").splitlines()):
        path = root / name
        if path.is_file():
            digest.update(name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


class Study:
    def __init__(self, source: Path, out: Path, tools: Path, source_label="F16", log=print):
        self.source, self.out, self.tools, self.log = source.resolve(strict=True), out.resolve(), tools.resolve(), log
        self.server, self.quantizer, self.ppl = (str(self.tools / name) for name in ("llama-server", "llama-quantize", "llama-perplexity"))
        self.label = source_label
        self.path = self.out / "study.json"
        self.out.mkdir(parents=True, exist_ok=True)
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {"phases": {}}

    def save(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.path)

    def done(self, phase):
        return self.state["phases"].get(phase, {}).get("status") == "completed"

    def finish(self, phase, **data):
        self.state["phases"][phase] = {"status": "completed", "finished_at": time.time(), **data}
        self.save()

    def variants(self) -> dict:
        variants = {self.label: inspect_gguf(self.source)}
        for job in self.state["phases"].get("convert", {}).get("jobs", []):
            variants[job["target"]] = inspect_gguf(job["artifact"]["path"])
            if variants[job["target"]]["sha256"] != job["artifact"]["sha256"]:
                raise ValueError(f"Artifact {job['target']} changed after conversion")
        return variants

    def session(self, artifact, name):
        return ServerSession(artifact, self.server, self.out / "logs" / f"{name}.log", context=2048, threads=4,
                             source_sha256=self.state["source"]["sha256"])

    # Phases -------------------------------------------------------------------------------------
    def provenance(self):
        if "source" in self.state:
            return
        root = Path(__file__).resolve().parents[2]
        source = inspect_gguf(self.source)
        source.pop("tensors", None)
        self.state.update(kind="real-run", study="quantization-study-v0.2", created_at=datetime.now(timezone.utc).isoformat(),
                          source=source, repository_commit=_git(root, "rev-parse", "HEAD"),
                          working_tree_diff_sha256=_tree_digest(root),
                          untracked_files=_git(root, "ls-files", "--others", "--exclude-standard").splitlines(),
                          toolchain_commit=_git(self.tools.parents[1], "rev-parse", "HEAD"),  # <llama.cpp>/build/bin
                          tools={name: sha256(self.tools / name) for name in ("llama-server", "llama-quantize", "llama-perplexity")},
                          hardware=hardware(), os=platform.platform(),
                          conditions="Run by the agent while VS Code and other user applications were open; see per-session flags.")
        self.save()

    def convert(self):
        if self.done("convert"):
            return
        jobs = []
        for target in TARGETS:
            directory = self.out / "quantization" / target
            directory.parent.mkdir(parents=True, exist_ok=True)
            manifest = directory / "manifest.json"
            if manifest.exists() and json.loads(manifest.read_text()).get("status") == "converted":
                job = json.loads(manifest.read_text())
            else:
                self.log(f"convert {target} (admission guard enforced)")
                try:
                    job = convert(self.source, directory, target, executable=self.quantizer)
                except ValueError as error:
                    if "Insufficient available RAM" not in str(error):
                        raise
                    snapshot = system_snapshot()
                    self.state["phases"]["convert"] = {
                        "status": "blocked", "stage": "quantization pre-flight admission", "target": target,
                        "reason": str(error), "jobs": jobs, "finished_at": time.time(),
                        "memory_at_rejection": {k: snapshot[k] for k in ("available_bytes", "swap", "pressure_level")}}
                    self.save()
                    self.log(f"  BLOCKED: {error}")
                    return
            memory = job.get("memory", {})
            limit = 1.5 * job["source"]["size_bytes"]
            job["admission_rule_check"] = {"measured_max_rss_bytes": memory.get("max_rss_bytes"), "rule_limit_bytes": int(limit),
                                           "violated": bool(memory.get("max_rss_bytes") and memory["max_rss_bytes"] > limit)}
            job.pop("source", None)
            job["artifact"].pop("tensors", None)
            jobs.append(job)
            self.log(f"  {target}: {job['artifact']['size_bytes'] / 2**20:.1f} MiB, peak RSS {(memory.get('max_rss_bytes') or 0) / 2**20:.0f} MiB, "
                     f"rule violated: {job['admission_rule_check']['violated']}")
        self.finish("convert", jobs=jobs)

    def validate(self):
        if self.done("validate"):
            return
        results = {}
        for label, artifact in self.variants().items():
            try:
                session = self.session(artifact, f"validate-{label}").start()
            except MemoryError as error:
                snapshot = system_snapshot()
                self.state["phases"]["validate"] = {
                    "status": "blocked", "stage": "runtime initialization pre-flight admission", "variant": label,
                    "reason": str(error), "results": results, "finished_at": time.time(),
                    "memory_at_rejection": {k: snapshot[k] for k in ("available_bytes", "swap", "pressure_level")}}
                self.save()
                self.log(f"  BLOCKED: {error}")
                return
            with session:
                output = session.generate(session.models()[0]["name"], VALIDATION_PROMPT,
                                          {"num_ctx": 2048, "num_predict": 32, "seed": 42, "temperature": 0})
                record = session.stop()
            results[label] = {"output": output["output"], "nonempty": bool(output["output"].strip()),
                              "metal": record["metal"], "load_seconds": record["load_seconds"],
                              "peak_memory_footprint_bytes": record["time_l"].get("peak_memory_footprint_bytes"),
                              "buffers_mib": record["buffers_mib"]}
            self.log(f"validate {label}: {output['output'].strip()[:40]!r} metal={record['metal']['verified']}")
            if not results[label]["nonempty"]:
                raise ValueError(f"{label} produced empty validation output")
        self.finish("validate", results=results, prompt=VALIDATION_PROMPT)

    def _evaluate(self, label, artifact, suite, split, name):
        try:
            session = self.session(artifact, name).start()
        except MemoryError as error:
            return {"path": None, "status": "blocked", "stage": "runtime initialization pre-flight admission",
                    "reason": str(error), "summary": {}, "by_category": {}, "sandbox": None}
        with session:
            def progress(done, total, row):
                if done % 20 == 0 or done == total:
                    self.log(f"  {name}: {done}/{total}")
            report = evaluate_dataset(session, session.models()[0]["name"], suite, split=split, context=2048, progress=progress)
            report["server_session"] = session.stop()
        report.pop("hardware", None)
        path = self.out / "quality" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=1) + "\n")
        return {"path": str(path.relative_to(self.out)), "status": report["status"], "summary": report["summary"],
                "by_category": report["by_category"], "sandbox": report.get("sandbox")}

    def dryrun(self):
        if self.done("dryrun"):
            return
        artifact = self.variants()[self.label]
        results = {suite: self._evaluate(self.label, artifact, suite, "screening", f"dryrun-{suite}-{self.label}")
                   for suite in ("reasoning-v1", "coding-v1")}
        self.finish("dryrun", results=results, note="Screening split on the baseline only; format check before held-out runs.")

    def benchmark(self, rounds=5, measured=4, cooldown=30):
        if self.done("benchmark"):
            return
        result = benchmark.run_benchmark(self.variants(), self.server, self.out / "benchmark", self.state["source"]["sha256"],
                                         rounds=rounds, measured=measured, cooldown=cooldown, log=self.log)
        (self.out / "benchmark" / "analysis.json").write_text(json.dumps(result["analysis"], indent=1) + "\n")
        self.finish("benchmark", design=result["design"], analysis=result["analysis"],
                     sessions_file="benchmark/sessions.jsonl", trials_file="benchmark/trials.csv")

    def quality(self):
        phase = self.state["phases"].setdefault("quality", {"status": "running", "runs": {}})
        if phase.get("status") == "completed":
            return
        for label, artifact in self.variants().items():
            for suite in ("reasoning-v1", "coding-v1"):
                key = f"{suite}-{label}"
                if key in phase["runs"] and phase["runs"][key]["status"] == "completed":
                    continue  # blocked or partial runs are retried on resume
                self.log(f"quality {key} (held-out)")
                phase["runs"][key] = self._evaluate(label, artifact, suite, "heldout", f"heldout-{key}")
                self.save()
        comparisons = {}
        for suite in ("reasoning-v1", "coding-v1"):
            base_run = phase["runs"][f"{suite}-{self.label}"]
            for target in self.targets():
                run = phase["runs"][f"{suite}-{target}"]
                if not base_run.get("path") or not run.get("path"):
                    comparisons[f"{suite}:{target}_vs_{self.label}"] = {"status": "insufficient_evidence", "reason": "A run did not execute"}
                    continue
                base = json.loads((self.out / base_run["path"]).read_text())
                candidate = json.loads((self.out / run["path"]).read_text())
                comparisons[f"{suite}:{target}_vs_{self.label}"] = compare_dataset_evaluations(base, candidate)
        statuses = {run["status"] for run in phase["runs"].values()}
        complete = statuses == {"completed"} and all(c.get("status") == "completed" for c in comparisons.values())
        phase.update(status="completed" if complete else "blocked" if statuses == {"blocked"} else "partial",
                     comparisons=comparisons, finished_at=time.time())
        self.save()

    def perplexity(self, chunks=20):
        if self.done("perplexity"):
            return
        directory = self.out / "perplexity"
        directory.mkdir(exist_ok=True)
        corpus = perplexity.build_corpus(directory / "corpus.txt")
        base_logits = directory / "base-logits.kld"  # large temporary file inside the ignored study directory
        results = {}
        for label, artifact in self.variants().items():
            self.log(f"perplexity {label}")
            results[label] = perplexity.run(self.ppl, artifact["path"], directory / "corpus.txt", directory / f"{label}.log",
                                            base_logits, base=label == self.label, chunks=chunks)
        base_logits.unlink(missing_ok=True)
        failed = [label for label, r in results.items()
                  if r["exit_code"] != 0 or (r.get("ppl") if label == self.label else r.get("mean_kld")) is None]
        for label in failed:
            results[label]["status"] = "failed"
        if failed:
            self.state["phases"]["perplexity"] = {"status": "failed", "failed_variants": failed, "corpus": corpus,
                                                  "results": results, "finished_at": time.time()}
            self.save()
            return
        self.finish("perplexity", corpus=corpus, results=results, base_logits="deleted after use (large temporary file)")

    def export(self, destination: Path) -> list[str]:
        """Copy curated, reviewable evidence (no weights, no large logits) into the repository."""
        import shutil
        destination.mkdir(parents=True, exist_ok=True)
        copied = []
        patterns = ["study.json", "benchmark/sessions.jsonl", "benchmark/trials.csv", "benchmark/analysis.json",
                    "quality/*.json", "perplexity/*.log", "quantization/*/manifest.json", "quantization/*/conversion.log"]
        for pattern in patterns:
            for path in sorted(self.out.glob(pattern)):
                target = destination / path.relative_to(self.out)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied.append(str(target))
        return copied

    def targets(self):
        return [job["target"] for job in self.state["phases"].get("convert", {}).get("jobs", [])]

    def run(self, phases=PHASES, **options):
        self.provenance()
        for phase in PHASES:
            if phase not in phases or phase == "report":
                continue
            self.log(f"== phase {phase}")
            started = system_snapshot()
            getattr(self, phase)(**{k: v for k, v in options.items() if k in {"benchmark": ("rounds", "measured", "cooldown")}.get(phase, ())})
            if phase not in self.state["phases"]:
                continue
            if self.state["phases"][phase].get("status") == "blocked" and phase != "convert":
                self.log(f"phase {phase} blocked; later phases need the blocked stage and were not run")
                break
            self.state["phases"][phase]["docker_vm_running_at_end"] = docker_vm_running()
            self.state["phases"][phase]["pressure_level_at_start"] = started["pressure_level"]
            self.save()
        return self.state
