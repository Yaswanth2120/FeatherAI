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
    args = parser.parse_args()
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
                result["summary"] = summarize(result["trials"])
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
