"""Daily operations — preflight, reports, scheduled ML retrain."""

import time

from config.settings import (
    AUTO_DAILY_REPORT,
    AUTO_TRAIN_ML,
    ML_RETRAIN_EVERY_CYCLES,
    PREFLIGHT_ON_START,
    USE_ML,
)
from core.preflight import Preflight
from research.daily_report import DailyReport


class DailyOps:
    def __init__(self):
        self._last_report_day = ""
        self._last_ml_cycle = 0

    def startup(self, live_mode: bool) -> bool:
        if not PREFLIGHT_ON_START and not live_mode:
            return True
        result = Preflight().run()
        Preflight().print_report(result)
        return result["ok"] or not live_mode

    def maybe_daily_report(self, equity: float, drawdown: float) -> None:
        if not AUTO_DAILY_REPORT:
            return
        day = time.strftime("%Y-%m-%d", time.gmtime())
        if day == self._last_report_day:
            return
        self._last_report_day = day
        try:
            report = DailyReport().generate(equity, drawdown, run_backtest=True)
            DailyReport().print_report(report)
        except Exception as e:
            print(f"[DailyReport] {e}")

    def maybe_retrain_ml(self, cycle: int) -> None:
        if not USE_ML or not AUTO_TRAIN_ML or ML_RETRAIN_EVERY_CYCLES <= 0:
            return
        if cycle - self._last_ml_cycle < ML_RETRAIN_EVERY_CYCLES:
            return
        self._last_ml_cycle = cycle
        try:
            from models.trainer import ModelTrainer
            r = ModelTrainer().train("BTC/USDT:USDT", epochs=150)
            print(f"[ML] Periodic retrain: {r}")
        except Exception as e:
            print(f"[ML] Retrain failed: {e}")
