import json
import time
import httpx


class Ollama:
    def __init__(self, url="http://127.0.0.1:11434", transport=None):
        self.client = httpx.Client(base_url=url, timeout=300, transport=transport)

    def close(self):
        self.client.close()

    def models(self):
        response = self.client.get("/api/tags")
        response.raise_for_status()
        return response.json()["models"]

    def version(self):
        response = self.client.get("/api/version")
        response.raise_for_status()
        return response.json()["version"]

    def generate(self, model, prompt, options):
        start = time.perf_counter()
        first = None
        output = []
        final = None
        with self.client.stream("POST", "/api/generate", json={
            "model": model, "prompt": prompt, "stream": True, "options": options,
            "think": False, "keep_alive": "5m"
        }) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                item = json.loads(line)
                if item.get("error"):
                    raise RuntimeError(item["error"])
                if item.get("response"):
                    if first is None:
                        first = time.perf_counter() - start
                    output.append(item["response"])
                if item.get("done"):
                    final = item
        if final is None:
            raise RuntimeError("Ollama stream ended without final statistics")
        return {"output": "".join(output), "ttft_seconds": first,
                "wall_seconds": time.perf_counter() - start, "runtime": final}
