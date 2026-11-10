import argparse
import csv
import json
import sys
import subprocess
from pathlib import Path
import httpx
from backend.backends.ollama import Ollama
from backend.benchmarks.runner import run, summarize, PROMPT
from backend.core.hardware import inspect
from backend.storage.database import Database
from backend.core.artifacts import inspect_gguf
from backend.quantization.gguf_adapter import convert, TARGETS


SANDBOX_PROBES = [
    ("correct", "def f(x):\n    return x * x", "passed"),
    ("incorrect", "def f(x):\n    return x + 1", "wrong_answer"),
    ("partial", "def f(x):\n    return 4 if x == 2 else -1", "partial_pass"),
    ("infinite_loop", "def f(x):\n    while True:\n        pass", "timeout"),
    ("memory_bomb", "def f(x):\n    a = []\n    while True:\n        a.append(bytearray(10**7))", "memory_limit"),
    ("fork_bomb", "import os\ndef f(x):\n    while True:\n        os.fork()", "runtime_error"),
    ("syntax_error", "def f(x) return x", "compile_error"),
    ("output_flood", "def f(x):\n    return 'x' * 10**6", "output_limit"),
]
ISOLATION_PROBE = ("import os, socket\ndef f(x):\n    try:\n        socket.create_connection(('1.1.1.1', 53), timeout=2); net = 'open'\n"
                   "    except OSError: net = 'blocked'\n    try:\n        open('/etc/feather', 'w'); fs = 'writable'\n"
                   "    except OSError: fs = 'read-only'\n    return [net, fs, os.getuid()]")


def sandbox_command(args) -> int:
    from backend.evaluation.sandbox import DockerExecutionBackend, PINNED_IMAGE, discover
    if args.action == "prepare":
        return subprocess.run(["docker", "pull", PINNED_IMAGE]).returncode
    backend = discover()
    status = backend.status()
    if args.action == "status" or status["status"] != "available":
        print(json.dumps(status, indent=2))
        return 0 if status["status"] == "available" else 1
    from backend.evaluation.coding import score_code
    from datetime import datetime, timezone
    task = {"function": "f", "cases": [([2], 4), ([3], 9), ([0], 0)]}
    rows = []
    for name, code, expected in SANDBOX_PROBES:
        result = score_code(code, task, sandbox=backend)
        rows.append({"probe": name, "expected_outcome": expected, "observed_outcome": result.get("outcome"),
                     "ok": result.get("outcome") == expected, "seconds": (result.get("execution") or {}).get("seconds")})
    isolation = score_code(ISOLATION_PROBE, {"function": "f", "cases": [([0], ["blocked", "read-only", 65534])]}, sandbox=backend)
    rows.append({"probe": "network_filesystem_user", "expected_outcome": "passed (network blocked, root read-only, uid 65534)",
                 "observed_outcome": isolation.get("outcome"), "ok": isolation.get("passed") is True})
    leftovers = subprocess.run(["docker", "ps", "-aq", "--filter", "name=feather-eval-"], capture_output=True, text=True).stdout.split()
    report = {"kind": "real-run", "created_at": datetime.now(timezone.utc).isoformat(), "sandbox": status, "probes": rows,
              "leftover_containers": len(leftovers), "status": "passed" if all(r["ok"] for r in rows) and not leftovers else "failed",
              "limitation": "Docker isolation is not an absolute security boundary; a compromised daemon, VM or kernel defeats it."}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0 if report["status"] == "passed" else 1


def automation_command(args) -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from automation.doctor import format_report, run_checks
    if args.action == "doctor":
        result = run_checks(root, do_probe=args.probe, probe_candidates=args.probe_candidates, select=args.select)
        print(json.dumps(result, indent=2) if args.json else format_report(result))
        return 0 if result["status"] == "ready" else 1
    from automation.codex_config import resolve_model
    from automation.selftest import run_selftest
    preflight = run_checks(root)
    if preflight["status"] == "blocked":
        print(format_report(preflight))
        return 1
    result = run_selftest(resolve_model(root)["model"], timeout=args.timeout)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description="Feather AI — local LLM laboratory")
    parser.add_argument("--db", default=".feather/experiments.sqlite3")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("hardware")
    models = commands.add_parser("models").add_subparsers(dest="action", required=True)
    models.add_parser("list")
    bench = commands.add_parser("benchmark").add_subparsers(dest="action", required=True)
    launch = bench.add_parser("run")
    launch.add_argument("--model", required=True)
    launch.add_argument("--backend", choices=["ollama"], default="ollama")
    launch.add_argument("--suite", choices=["coding-small"], default="coding-small", help="Coding prompt timing only; does not execute or score code")
    for name, default in [("repeats", 5), ("warmups", 1), ("context", 2048), ("tokens", 128), ("seed", 42)]:
        launch.add_argument("--" + name, type=int, default=default)
    launch.add_argument("--prompt-file", type=Path)
    report = bench.add_parser("report")
    report.add_argument("--latest", action="store_true")
    report.add_argument("--id")
    report.add_argument("--output", type=Path)
    report.add_argument("--format", choices=["json", "csv"], default="json")
    artifact = commands.add_parser("artifact")
    artifact.add_argument("path", type=Path)
    quant = commands.add_parser("quantize")
    quant.add_argument("--source", type=Path, required=True)
    quant.add_argument("--output-dir", type=Path, required=True)
    quant.add_argument("--target", choices=TARGETS, default="Q4_K_M")
    quant.add_argument("--executable", default="llama-quantize")
    quant.add_argument("--timeout", type=int, default=1800)
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model-root", type=Path, default=Path("models"))
    serve.add_argument("--quantizer", default="llama-quantize")
    serve.add_argument("--server", default="llama-server")
    study = commands.add_parser("study")
    study.add_argument("--source", type=Path, required=True)
    study.add_argument("--output-dir", type=Path, required=True)
    study.add_argument("--quantizer", required=True)
    study.add_argument("--server", required=True)
    study.add_argument("--repeats", type=int, default=5)
    automation = commands.add_parser("automation", help="Codex automation preflight and self-test").add_subparsers(dest="action", required=True)
    doctor = automation.add_parser("doctor", help="Validate CLI, auth readiness, model, repository, queue and paths")
    doctor.add_argument("--probe", action="store_true", help="Send one small read-only request to verify the model (uses account quota)")
    doctor.add_argument("--probe-candidates", action="store_true", help="With --probe, try listed cached models until one is accepted")
    doctor.add_argument("--select", action="store_true", help="With --probe, record the accepted model for automation")
    doctor.add_argument("--json", action="store_true")
    selftest = automation.add_parser("selftest", help="Run two dependent tasks through live Codex in a throwaway repository")
    selftest.add_argument("--timeout", type=int, default=600)
    quant = commands.add_parser("quant-study", help="v0.2 reproducible quantization study (resumable phases)")
    quant.add_argument("--source", type=Path, required=True)
    quant.add_argument("--out", type=Path, required=True, help="Working directory (keep it under .feather/)")
    quant.add_argument("--tools", type=Path, default=Path(".feather/tools/llama.cpp/build/bin"))
    quant.add_argument("--phases", default="convert,validate,dryrun,benchmark,quality,perplexity")
    quant.add_argument("--rounds", type=int, default=5)
    quant.add_argument("--measured", type=int, default=4)
    quant.add_argument("--cooldown", type=int, default=30)
    quant.add_argument("--export", type=Path, help="Copy curated evidence to this directory after the phases")
    e2e = commands.add_parser("e2e", help="One traceable end-to-end workflow through the local API (task08)")
    e2e.add_argument("--source", type=Path, required=True)
    e2e.add_argument("--constraints", type=Path, default=Path("experiments/e2e/recommendation-constraints.json"))
    e2e.add_argument("--tools", type=Path, default=Path(".feather/tools/llama.cpp/build/bin"))
    e2e.add_argument("--out", type=Path, default=Path("experiments/e2e"))
    sandbox = commands.add_parser("sandbox", help="Coding-evaluation isolation").add_subparsers(dest="action", required=True)
    sandbox.add_parser("status", help="Report whether isolated execution is available")
    sandbox.add_parser("prepare", help="Explicitly pull the pinned sandbox image (about 20 MB)")
    sandbox_check = sandbox.add_parser("selftest", help="Run isolation acceptance probes inside the sandbox")
    sandbox_check.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "sandbox":
        sys.exit(sandbox_command(args))
    if args.command == "e2e":
        from backend.e2e import Workflow, summary_evidence
        workflow = Workflow(args.source, args.constraints, args.tools, args.out, log=lambda m: print(m, flush=True))
        try:
            record = workflow.run(json.loads(args.constraints.read_text()))
        finally:
            evidence = summary_evidence(workflow.out)
            Path("experiments/e2e-evidence-v0.2.json").write_text(json.dumps(evidence, indent=2) + "\n")
            print(json.dumps(evidence, indent=2))
        sys.exit(0 if record["status"] == "completed" else 1)
    if args.command == "quant-study":
        from backend.study.run import Study
        study = Study(args.source, args.out, args.tools, log=lambda m: print(m, flush=True))
        state = study.run([p.strip() for p in args.phases.split(",") if p.strip()],
                          rounds=args.rounds, measured=args.measured, cooldown=args.cooldown)
        if args.export:
            from backend.study.report import export_tables
            print("\n".join(study.export(args.export)))
            print("\n".join(str(p) for p in export_tables(args.export / "study.json", args.export)))
        print(json.dumps({k: v.get("status") for k, v in state["phases"].items()}, indent=2))
        sys.exit(0)
    if args.command == "automation":
        sys.exit(automation_command(args))
    backend = None
    try:
        if args.command == "serve":
            import uvicorn
            from backend.api.app import create_app
            app = create_app(data_dir=Path(args.db).parent, model_root=args.model_root, quantizer=args.quantizer, server_executable=args.server)
            uvicorn.run(app, host="127.0.0.1", port=args.port)
            return
        if args.command == "study":
            from backend.lab import study
            result = study(args.source, args.output_dir, args.quantizer, args.server, args.db, args.repeats)
        elif args.command == "hardware":
            result = inspect()
        elif args.command == "artifact":
            result = inspect_gguf(args.path)
        elif args.command == "quantize":
            result = convert(args.source, args.output_dir, args.target, args.executable, args.timeout)
        elif args.command == "models":
            backend = Ollama(args.ollama_url)
            result = backend.models()
        else:
            db = Database(args.db)
            if args.action == "run":
                backend = Ollama(args.ollama_url)
                result = run(backend, db, args.model, args.repeats, args.warmups, args.context,
                             args.tokens, args.seed, args.prompt_file.read_text() if args.prompt_file else PROMPT)
            else:
                result = db.report(args.id)
                result["summary"] = summarize(result["trials"], result["metadata"])
                if args.format == "csv":
                    stream = args.output.open("w", newline="") if args.output else sys.stdout
                    try:
                        fields = ["ordinal", "warmup", "wall_seconds", "ttft_seconds", "decode_tokens_per_second", "prefill_tokens_per_second", "error"]
                        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
                        writer.writeheader()
                        writer.writerows(result["trials"])
                    finally:
                        if args.output:
                            stream.close()
                    return
                if args.output:
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
                    return
        print(json.dumps(result, indent=2))
        if args.command == "benchmark" and args.action == "run" and result["summary"]["failed_trials"]:
            sys.exit(1)
    except (httpx.HTTPError, ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"feather: {error}\n")
    finally:
        if backend:
            backend.close()


if __name__ == "__main__":
    main()
