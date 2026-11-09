"""One traceable end-to-end workflow through the real local API (task08).

import -> quantization jobs -> generation validation -> benchmark jobs -> sandboxed coding
evaluation jobs -> recommendation -> exports, all under one workflow_id with hash chaining.
Every stage is recorded, including failures; nothing is reused from earlier runs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

import httpx

from backend.backends.llama_cpp import LlamaCpp
from backend.core.artifacts import inspect_gguf, sha256
from backend.core.hardware import inspect as hardware
from backend.evaluation.sandbox import discover

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("Q8_0", "Q4_K_M")
SETTINGS = {"context": 2048, "seed": 42, "temperature": 0, "benchmark_repeats": 5, "benchmark_warmups": 1,
            "benchmark_tokens": 128, "coding_tokens": 384, "quality_suite": "coding-v1"}
VALIDATION_PROMPT = "What is 2 + 2? Answer with only the number."


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Workflow:
    def __init__(self, source: Path, constraints: Path, tools: Path, out_root: Path, log=print):
        self.id = str(uuid.uuid4())
        self.source, self.constraints_path = source.resolve(strict=True), constraints.resolve(strict=True)
        self.tools, self.log = tools.resolve(), log
        self.out = out_root.resolve() / self.id
        self.out.mkdir(parents=True)
        self.data_dir = ROOT / ".feather" / "e2e" / self.id
        self.model_root = self.source.parent
        self.record = {"kind": "real-run", "workflow_id": self.id, "started_at": datetime.now(timezone.utc).isoformat(),
                       "status": "running", "settings": SETTINGS, "stages": {},
                       "constraints_file": str(self.constraints_path.relative_to(ROOT)),
                       "constraints_sha256": _digest(self.constraints_path)}

    # Bookkeeping -------------------------------------------------------------------------------
    def save(self):
        temporary = self.out / "workflow.tmp"
        temporary.write_text(json.dumps(self.record, indent=1, allow_nan=False) + "\n")
        temporary.replace(self.out / "workflow.json")

    def stage(self, name, **data):
        self.record["stages"][name] = {"workflow_id": self.id, "recorded_at": time.time(), **data}
        self.save()

    def keep(self, name: str, payload) -> str:
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1) + "\n") if not isinstance(payload, (bytes, str)) else \
            (path.write_bytes(payload) if isinstance(payload, bytes) else path.write_text(payload))
        return str(path.relative_to(self.out))

    # Server --------------------------------------------------------------------------------------
    def start_api(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.data_dir.mkdir(parents=True)
        command = [sys.executable, "-m", "backend.cli", "--db", str(self.data_dir / "experiments.sqlite3"), "serve",
                   "--port", str(port), "--model-root", str(self.model_root),
                   "--quantizer", str(self.tools / "llama-quantize"), "--server", str(self.tools / "llama-server")]
        self.api_log = (self.out / "api-server.log").open("w")
        self.api = subprocess.Popen(command, cwd=ROOT, stdout=self.api_log, stderr=subprocess.STDOUT)
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60)
        for _ in range(600):
            if self.api.poll() is not None:
                raise RuntimeError(f"Local API exited with code {self.api.returncode}; see api-server.log")
            try:
                if self.client.get("/api/health").status_code == 200:
                    return
            except httpx.TransportError:
                time.sleep(.1)
        raise RuntimeError("Local API did not start")

    def stop_api(self):
        if getattr(self, "api", None) and self.api.poll() is None:
            self.api.terminate()
            try:
                self.api.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.api.kill()
        if getattr(self, "api_log", None):
            self.api_log.close()

    def wait(self, job_id: str, timeout=3600) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.client.get(f"/api/jobs/{job_id}").json()
            if job["status"] not in {"queued", "running", "pending"}:
                self.keep(f"jobs/{job_id}.json", job)
                return job
            time.sleep(1)
        raise TimeoutError(f"Job {job_id} did not finish")

    # Stages --------------------------------------------------------------------------------------
    def run(self, constraints: dict) -> dict:
        try:
            # Provenance first: before this run writes any file into the repository.
            from backend.study.run import _tree_digest
            self.record["environment"] = {"repository_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                                          "repository_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()),
                                          "working_tree_diff_sha256": _tree_digest(ROOT),
                                          "untracked_files": subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True).stdout.split(),
                                          "toolchain_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.tools.parents[1], capture_output=True, text=True).stdout.strip(),
                                          "hardware": {k: v for k, v in hardware().items() if k != "system_swap"}}
            self.start_api()
            source = inspect_gguf(self.source)
            source.pop("tensors", None)
            if not source["higher_precision"]:
                raise ValueError("Source must contain only F32/F16/BF16 tensors")
            self.stage("import", status="completed", source=source)
            self.log(f"[{self.id}] import {source['sha256'][:12]} {source['size_bytes']} bytes")

            sandbox = discover().status()
            self.stage("preflight", status="completed" if sandbox["status"] == "available" else "blocked", sandbox=sandbox,
                       available_memory_bytes=hardware()["available_ram_bytes"])
            if sandbox["status"] != "available":
                raise RuntimeError(f"Sandbox unavailable: {sandbox.get('reason')}")

            quantized = {}
            for target in TARGETS:
                job = self.client.post("/api/jobs/quantization", json={"source": self.source.name, "target": target}).json()
                job = self.wait(job["id"])
                result = job.get("result") or {}
                artifact = result.get("artifact") or {}
                quantized[target] = {"job_id": job["id"], "status": job["status"], "error": job.get("error"),
                                     "source_sha256": (result.get("source") or {}).get("sha256"),
                                     "artifact_sha256": artifact.get("sha256"), "artifact_path": artifact.get("path"),
                                     "artifact_size_bytes": artifact.get("size_bytes"), "memory": result.get("memory"),
                                     "admission": {k: result.get(k) for k in ("memory_admission_estimate_bytes", "memory_admission_rule", "available_ram_bytes")}}
                self.log(f"  quantize {target}: {job['status']} {str(artifact.get('sha256'))[:12]}")
            self.stage("quantization", status="completed" if all(q["status"] == "completed" for q in quantized.values()) else "failed",
                       artifacts=quantized)
            if self.record["stages"]["quantization"]["status"] != "completed":
                raise RuntimeError("Quantization failed")

            validations = {}
            for target, q in quantized.items():
                if sha256(q["artifact_path"]) != q["artifact_sha256"]:
                    raise ValueError(f"{target} artifact changed after conversion")
                with LlamaCpp(q["artifact_path"], executable=str(self.tools / "llama-server"), context=SETTINGS["context"],
                              source_sha256=source["sha256"], log_path=self.out / f"validate-{target}.log") as backend:
                    output = backend.generate(backend.models()[0]["name"], VALIDATION_PROMPT,
                                              {"num_ctx": SETTINGS["context"], "num_predict": 32, "seed": SETTINGS["seed"], "temperature": 0})
                    version = backend.version()
                validations[target] = {"artifact_sha256": q["artifact_sha256"], "prompt": VALIDATION_PROMPT, "output": output["output"],
                                       "nonempty": bool(output["output"].strip()), "runtime_version": version,
                                       "criterion": "runtime compatibility: the exact artifact loads in llama-server and returns non-empty text; correctness is measured by the evaluation stage",
                                       "contains_expected_answer_4": "4" in output["output"]}
                self.log(f"  validate {target}: {output['output'].strip()[:30]!r}")
            self.stage("generation_validation", status="completed" if all(v["nonempty"] for v in validations.values()) else "failed",
                       results=validations)

            benchmarks = {}
            for target, q in quantized.items():
                job = self.client.post("/api/jobs/benchmark", json={
                    "backend": "llama.cpp", "artifact_job_id": q["job_id"], "model": "managed", "repeats": SETTINGS["benchmark_repeats"],
                    "warmups": SETTINGS["benchmark_warmups"], "context": SETTINGS["context"], "tokens": SETTINGS["benchmark_tokens"],
                    "seed": SETTINGS["seed"]}).json()
                job = self.wait(job["id"])
                report = job.get("result") or {}
                metadata = report.get("metadata") or {}
                benchmarks[target] = {"job_id": job["id"], "status": job["status"], "error": job.get("error"), "experiment_id": report.get("id"),
                                      "source_sha256": metadata.get("source_sha256"),
                                      "artifact_sha256": (metadata.get("model") or {}).get("artifact_sha256"),
                                      "settings_sha256": metadata.get("settings_sha256"), "summary": report.get("summary")}
                self.log(f"  benchmark {target}: {job['status']} decode median {((report.get('summary') or {}).get('decode_tokens_per_second') or {}).get('median')}")
            self.stage("benchmark", status="completed" if all(b["status"] == "completed" for b in benchmarks.values()) else "failed",
                       results=benchmarks)

            evaluations = {}
            for target, q in quantized.items():
                job = self.client.post("/api/jobs/evaluation", json={
                    "backend": "llama.cpp", "artifact_job_id": q["job_id"], "model": "managed", "suite": SETTINGS["quality_suite"],
                    "context": SETTINGS["context"], "tokens": SETTINGS["coding_tokens"], "seed": SETTINGS["seed"]}).json()
                job = self.wait(job["id"])
                report = job.get("result") or {}
                evaluations[target] = {"job_id": job["id"], "status": job["status"], "error": job.get("error"), "evaluation_id": report.get("id"),
                                       "source_sha256": report.get("source_sha256"),
                                       "artifact_sha256": (report.get("model") or {}).get("artifact_sha256"),
                                       "dataset_sha256": report.get("dataset_sha256"), "split": report.get("split"),
                                       "settings_sha256": report.get("settings_sha256"), "sandbox": report.get("sandbox"),
                                       "summary": report.get("summary")}
                summary = report.get("summary") or {}
                self.log(f"  coding {target}: {job['status']} {summary.get('passed')}/{summary.get('items')}")
            self.stage("evaluation", status="completed" if all(e["status"] == "completed" for e in evaluations.values()) else "failed",
                       results=evaluations)

            available = self.client.get("/api/hardware").json()["available_ram_bytes"]
            request = {"experiment_ids": [b["experiment_id"] for b in benchmarks.values()], "available_memory_bytes": available,
                       "max_memory_bytes": constraints["max_memory_bytes"], "min_quality": constraints["min_quality"],
                       "max_latency_seconds": constraints["max_latency_seconds"], "objective": constraints["objective"],
                       "quality_suite": constraints["quality_suite"], "quality_split": constraints["quality_split"]}
            response = self.client.post("/api/recommendations", json=request)
            recommendation = response.json()
            self.keep("recommendation.json", {"request": request, "response": recommendation, "http_status": response.status_code})
            self.stage("recommendation", status="completed" if response.status_code == 200 and recommendation.get("status") in
                       set(constraints["expected_valid_outcomes"]) else "failed",
                       request=request, outcome=recommendation.get("status"), selected_id=recommendation.get("selected_id"),
                       eligible=[c["id"] for c in recommendation.get("eligible", [])],
                       rejected=recommendation.get("rejected"), http_status=response.status_code)
            self.log(f"  recommendation: {recommendation.get('status')}")

            files = {}
            for target, b in benchmarks.items():
                for fmt in ("json", "csv"):
                    response = self.client.get(f"/api/experiments/{b['experiment_id']}/export", params={"format": fmt})
                    name = f"exports/benchmark-{target}.{fmt}"
                    self.keep(name, response.content)
                    files[name] = {"sha256": hashlib.sha256(response.content).hexdigest(), "http_status": response.status_code,
                                   "experiment_id": b["experiment_id"]}
            for target, e in evaluations.items():
                evaluation = next((item for item in self.client.get("/api/evaluations").json() if item.get("id") == e["evaluation_id"]), None)
                name = f"exports/evaluation-{target}.json"
                content = json.dumps(evaluation, indent=1).encode()
                self.keep(name, content)
                files[name] = {"sha256": hashlib.sha256(content).hexdigest(), "http_status": 200 if evaluation else 404,
                               "evaluation_id": e["evaluation_id"]}
            self.stage("export", status="completed" if all(f["http_status"] == 200 for f in files.values()) else "failed", files=files)
            statuses = [stage["status"] for stage in self.record["stages"].values()]
            self.record["status"] = "completed" if set(statuses) == {"completed"} else "failed"
        except Exception as error:
            self.record.update(status="failed", error=f"{type(error).__name__}: {error}"[:1000])
            raise
        finally:
            self.record["finished_at"] = datetime.now(timezone.utc).isoformat()
            self.save()
            self.stop_api()
        return self.record


def summary_evidence(workflow_dir: Path) -> dict:
    record = json.loads((workflow_dir / "workflow.json").read_text())
    stages = record["stages"]
    done = lambda name: stages.get(name, {}).get("status") == "completed"
    return {"kind": "real-run", "status": record["status"], "workflow": str((workflow_dir / "workflow.json").relative_to(ROOT)),
            "workflow_id": record["workflow_id"],
            "imported": done("import"), "quantized": done("quantization"), "generation_validated": done("generation_validation"),
            "benchmarked": done("benchmark"), "evaluated": done("evaluation"), "coding_evaluated": done("evaluation"),
            "recommended": done("recommendation"), "exported": done("export"),
            "recommendation_outcome": stages.get("recommendation", {}).get("outcome")}
