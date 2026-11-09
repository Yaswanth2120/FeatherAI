import json
import sqlite3
from pathlib import Path


class Database:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, metadata TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS trials (experiment_id TEXT, ordinal INTEGER, result TEXT NOT NULL, PRIMARY KEY(experiment_id, ordinal))")

    def start(self, identifier, metadata):
        with self.connection:
            self.connection.execute("INSERT INTO experiments VALUES (?, ?)", (identifier, json.dumps(metadata)))

    def save(self, identifier, ordinal, result):
        with self.connection:
            self.connection.execute("INSERT INTO trials VALUES (?, ?, ?)", (identifier, ordinal, json.dumps(result)))

    def report(self, identifier=None):
        row = self.connection.execute("SELECT id, metadata FROM experiments " +
            ("WHERE id=?" if identifier else "ORDER BY rowid DESC LIMIT 1"),
            (identifier,) if identifier else ()).fetchone()
        if row is None:
            raise ValueError("No saved experiment found")
        trials = self.connection.execute("SELECT result FROM trials WHERE experiment_id=? ORDER BY ordinal", (row[0],)).fetchall()
        return {"id": row[0], "metadata": json.loads(row[1]), "trials": [json.loads(t[0]) for t in trials]}
