"""Persistent jobs with one heavy worker across API processes."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import psutil

TERMINAL = {'completed', 'failed', 'cancelled', 'interrupted', 'partial', 'unsupported', 'cleanup_required'}


class Jobs:
    def __init__(self, root: Path, config: dict):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.lock_file = None
        self.stop_event = threading.Event()
        self.thread = None
        self.process = None
        with self.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
                payload TEXT NOT NULL, result TEXT, error TEXT,
                created_at REAL NOT NULL, updated_at REAL NOT NULL,
                cancel_requested INTEGER NOT NULL DEFAULT 0, pid INTEGER)''')

    def connect(self):
        db = sqlite3.connect(self.root / 'jobs.sqlite3', timeout=10)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def decode(row):
        if row is None:
            raise KeyError('Job not found')
        result = dict(row)
        result['payload'] = json.loads(result['payload'])
        result['result'] = json.loads(result['result']) if result['result'] else None
        return result

    def list(self):
        with self.connect() as db:
            return [self.decode(row) for row in db.execute('SELECT * FROM jobs ORDER BY created_at DESC LIMIT 200')]

    def get(self, identifier):
        with self.connect() as db:
            return self.decode(db.execute('SELECT * FROM jobs WHERE id=?', (identifier,)).fetchone())

    def add(self, kind, payload):
        identifier = str(uuid.uuid4())
        now = time.time()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0] >= 100:
                raise ValueError('Job queue is full')
            db.execute('INSERT INTO jobs(id,kind,status,payload,created_at,updated_at) VALUES (?,?,?,?,?,?)',
                       (identifier, kind, 'queued', json.dumps(payload), now, now))
        return self.get(identifier)

    def update(self, identifier, status, result=None, error=None, pid=None):
        with self.connect() as db:
            db.execute('UPDATE jobs SET status=?,result=?,error=?,pid=?,updated_at=? WHERE id=?',
                       (status, json.dumps(result) if result is not None else None, error, pid, time.time(), identifier))

    def cancel(self, identifier):
        job = self.get(identifier)
        if job['status'] in TERMINAL:
            return job
        with self.connect() as db:
            db.execute("UPDATE jobs SET cancel_requested=1,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,updated_at=? WHERE id=? AND status NOT IN ('completed','failed','cancelled','interrupted','partial','unsupported','cleanup_required')",
                       (time.time(), identifier))
        return self.get(identifier)

    def start(self):
        self.lock_file = (self.root / 'worker.lock').open('a+')
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock_file.close()
            self.lock_file = None
            raise RuntimeError('Another Feather API worker owns this data directory')
        # The old worker monitors its parent, but also recover rows after abrupt API termination.
        for job in self.list():
            if job['status'] in {'running', 'cleanup_required'}:
                if job.get('pid'):
                    try:
                        stale = psutil.Process(job['pid'])
                        command = stale.cmdline()
                        expected = str(self.root / 'jobs' / job['id'] / 'request.json')
                        if 'backend.api.worker' in command and expected in command:
                            os.killpg(stale.pid, signal.SIGKILL)
                    except (psutil.Error, ProcessLookupError):
                        pass
                cleanup_error = self.cleanup(job)
                if cleanup_error:
                    self.update(job['id'], 'cleanup_required', result=job['result'], error=cleanup_error)
                    self.stop()
                    raise RuntimeError(cleanup_error)
                self.update(job['id'], 'interrupted', result=job['result'], error='API stopped before job completion; submit a new job')
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.loop, name='feather-jobs', daemon=True)
        self.thread.start()

    @staticmethod
    def kill(process):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=8)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=18)
            if self.thread.is_alive() and self.process:
                self.kill(self.process)
                self.thread.join(timeout=3)
        if self.lock_file:
            fcntl.flock(self.lock_file, fcntl.LOCK_UN)
            self.lock_file.close()
            self.lock_file = None

    def loop(self):
        while not self.stop_event.is_set():
            with self.connect() as db:
                row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                self.stop_event.wait(.1)
                continue
            job = self.decode(row)
            # Recheck cancellation atomically before claiming.
            with self.connect() as db:
                claimed = db.execute("UPDATE jobs SET status='running',updated_at=? WHERE id=? AND status='queued' AND cancel_requested=0",
                                     (time.time(), job['id'])).rowcount
            if claimed:
                try:
                    self.execute(job)
                except Exception as error:
                    self.update(job['id'], 'failed', error=f'{type(error).__name__}: {error}')

    @staticmethod
    def cleanup(job):
        if job['kind'] != 'evaluation' or job['payload'].get('suite') != 'coding-small':
            return None
        try:
            result = job.get('result')
            if isinstance(result, dict) and result.get('status') == 'unsupported':
                return None
            from backend.evaluation.coding import cleanup_job
            cleanup_job(job['id'])
        except Exception as error:
            return f'Container cleanup not confirmed: {type(error).__name__}: {error}'
        return None

    @staticmethod
    def result_status(job, result):
        if job['kind'] == 'evaluation':
            return result.get('status') if result.get('status') in {'completed', 'partial', 'unsupported'} else 'failed'
        if job['kind'] == 'benchmark':
            summary = result.get('summary', {})
            if summary.get('failed_trials', 0):
                return 'partial' if summary.get('successful_trials', 0) else 'failed'
        return 'completed'

    def execute(self, job):
        folder = self.root / 'jobs' / job['id']
        folder.mkdir(parents=True)
        request = {'id': job['id'], 'kind': job['kind'], 'payload': job['payload'],
                   'config': self.config, 'root': str(self.root), 'parent_pid': os.getpid()}
        request_path = folder / 'request.json'
        request_path.write_text(json.dumps(request))
        with (folder / 'worker.log').open('wb') as log:
            process = subprocess.Popen([sys.executable, '-m', 'backend.api.worker', str(request_path)],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.process = process
            self.update(job['id'], 'running', pid=process.pid)
            deadline = time.monotonic() + self.config.get('job_timeout', 1800)
            status = None
            while process.poll() is None:
                if self.stop_event.wait(.1):
                    status = 'interrupted'
                elif self.get(job['id'])['cancel_requested']:
                    status = 'cancelled'
                elif time.monotonic() > deadline:
                    status = 'failed'
                if status:
                    self.kill(process)
                    self.update(job['id'], status, error='Job timeout' if status == 'failed' else f'Job {status}')
                    break
            if status is None:
                if self.get(job['id'])['cancel_requested']:
                    self.update(job['id'], 'cancelled', error='Job cancelled')
                elif process.returncode == 0 and (folder / 'result.json').is_file():
                    result = json.loads((folder / 'result.json').read_text())
                    status = self.result_status(job, result)
                    self.update(job['id'], status, result=result,
                                error='Some or all requested measurements were unavailable or failed' if status != 'completed' else None)
                else:
                    error_path = folder / 'error.json'
                    error = json.loads(error_path.read_text()).get('error') if error_path.exists() else f'Worker exited {process.returncode}; inspect worker.log'
                    self.update(job['id'], 'failed', error=error)
            # A worker that exited unexpectedly may have left subprocesses behind.
            # All children share its dedicated group; do not leave inference running.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            cleanup_error = self.cleanup(self.get(job['id']))
            if cleanup_error:
                current = self.get(job['id'])
                self.update(job['id'], 'cleanup_required', result=current['result'], error=cleanup_error)
                self.stop_event.set()
            self.process = None
