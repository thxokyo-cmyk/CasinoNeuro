"""
SQLite database for storing roulette spin history.
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Optional, Tuple

from logger import log


class SpinDatabase:
    def __init__(self, db_path: str = "roulette_spins.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                number TEXT NOT NULL,
                color TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source TEXT DEFAULT 'auto',
                session_id TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON spins(timestamp)
        """)

        conn.commit()
        conn.close()
        log.info("[DB] Database initialized: " + self.db_path)

    def add_spin(self, number: str, color: str, source: str = "auto", session_id: str = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO spins (number, color, source, session_id)
            VALUES (?, ?, ?, ?)
        """, (number, color, source, session_id))

        conn.commit()
        conn.close()

    def remove_last_spin(self) -> bool:
        """Remove the most recent spin. Returns True if successful."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM spins ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()

        if row:
            cursor.execute("DELETE FROM spins WHERE id = ?", (row[0],))
            conn.commit()
            conn.close()
            log.info("[DB] Removed spin id={}".format(row[0]))
            return True

        conn.close()
        return False

    def get_recent_numbers(self, limit: int = 50) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT number FROM spins
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        results = cursor.fetchall()
        conn.close()

        return [r[0] for r in reversed(results)]

    def get_all_numbers(self) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT number FROM spins
            ORDER BY timestamp ASC
        """)

        results = cursor.fetchall()
        conn.close()

        return [r[0] for r in results]

    def get_recent_spins(self, limit: int = 50) -> List[Tuple]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT number, color, timestamp, source FROM spins
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        results = cursor.fetchall()
        conn.close()

        return list(reversed(results))

    def get_total_spins(self) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM spins")
        count = cursor.fetchone()[0]

        conn.close()
        return count

    def get_statistics(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM spins")
        total = cursor.fetchone()[0]

        cursor.execute("""
            SELECT color, COUNT(*) as cnt FROM spins
            GROUP BY color
        """)
        colors = {r[0]: r[1] for r in cursor.fetchall()}

        cursor.execute("""
            SELECT number, COUNT(*) as cnt FROM spins
            GROUP BY number
            ORDER BY cnt DESC
            LIMIT 10
        """)
        hot_numbers = [(r[0], r[1]) for r in cursor.fetchall()]

        conn.close()

        return {
            "total": total,
            "colors": colors,
            "hot_numbers": hot_numbers
        }

    def clear_all(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM spins")
        conn.commit()
        conn.close()
        log.info("[DB] All spins cleared")