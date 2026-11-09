"""Persist evaluation reports separately from performance trials in the local database."""
import json
import sqlite3
from pathlib import Path


class EvaluationStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS evaluations (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, report TEXT NOT NULL)')

    def save(self, report):
        with sqlite3.connect(self.path) as db:
            db.execute('INSERT INTO evaluations VALUES (?,?,?)', (report['id'], report['created_at'], json.dumps(report, allow_nan=False)))

    def list(self):
        with sqlite3.connect(self.path) as db:
            return [json.loads(row[0]) for row in db.execute('SELECT report FROM evaluations ORDER BY created_at DESC LIMIT 200')]
