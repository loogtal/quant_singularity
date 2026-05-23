# research/trade_logger.py

import sqlite3
from pathlib import Path
from datetime import datetime


class TradeLogger:

    def __init__(self):

        # root project
        self.project_root = Path(__file__).resolve().parent.parent

        # storage directory
        self.storage_dir = self.project_root / "storage"

        # create folder automatically
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # db path
        self.db_path = self.storage_dir / "trades.db"

        # sqlite connection
        self.conn = sqlite3.connect(str(self.db_path))

        self.create_tables()

    def create_tables(self):

        cursor = self.conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            side TEXT,
            entry_price REAL,
            exit_price REAL,
            qty REAL,
            pnl REAL,
            strategy TEXT,
            regime TEXT
        )
        """)

        self.conn.commit()

    def log_trade(self, trade: dict):

        cursor = self.conn.cursor()

        cursor.execute("""
        INSERT INTO trades (
            timestamp,
            symbol,
            side,
            entry_price,
            exit_price,
            qty,
            pnl,
            strategy,
            regime
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(datetime.utcnow()),
            trade.get("symbol"),
            trade.get("side"),
            trade.get("entry_price"),
            trade.get("exit_price"),
            trade.get("qty"),
            trade.get("pnl"),
            trade.get("strategy"),
            trade.get("regime")
        ))

        self.conn.commit()

    def fetch_all(self):

        cursor = self.conn.cursor()

        cursor.execute("SELECT * FROM trades")

        return cursor.fetchall()

    def close(self):

        self.conn.close()
