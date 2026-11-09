"""Sample owned runtime process RSS; never substitute system usage for model memory."""
import threading
import psutil


class ProcessMonitor:
    def __init__(self, pid, interval=.05):
        self.pid = pid
        self.interval = interval
        self.peak = None
        self.samples = 0
        self.stop_event = threading.Event()
        self.thread = None

    def sample(self):
        try:
            process = psutil.Process(self.pid)
            memory = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    memory += child.memory_info().rss
                except psutil.Error:
                    pass
            self.peak = max(self.peak or 0, memory)
            self.samples += 1
        except psutil.Error:
            pass

    def __enter__(self):
        self.sample()
        def loop():
            while not self.stop_event.wait(self.interval):
                self.sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.sample()
        self.stop_event.set()
        self.thread.join(timeout=1)
