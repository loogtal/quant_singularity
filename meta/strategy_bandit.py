"""Thompson Sampling multi-armed bandit for autonomous strategy/mode selection per regime."""

import json
import random

from config.settings import STORAGE_DIR

BANDIT_FILE = STORAGE_DIR / "bandit_state.json"

PASSIVE_ARMS = ["long", "short", "skip"]
ACTIVE_ARMS  = ["momentum", "mean_reversion", "funding_arb", "vwap_reversal", "breakout"]

# Minimum trials before bandit can override the regime recommendation
WARMUP_TRIALS = 8


class ThompsonBandit:
    """
    Per-regime Thompson Sampling bandit.
    Each arm has Beta(alpha, beta) posterior where alpha=wins+1, beta=losses+1.
    Sample from each arm's distribution, pick the highest — exploration is automatic.
    """

    def __init__(self, arms: list[str], name: str):
        self.name = name
        self.arms = arms
        # {regime: {arm: [wins, losses]}}
        self._stats: dict[str, dict[str, list[int]]] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not BANDIT_FILE.exists():
            return
        try:
            data = json.loads(BANDIT_FILE.read_text())
            self._stats = data.get(self.name, {})
        except Exception:
            pass

    def _save(self) -> None:
        try:
            existing: dict = {}
            if BANDIT_FILE.exists():
                try:
                    existing = json.loads(BANDIT_FILE.read_text())
                except Exception:
                    pass
            existing[self.name] = self._stats
            BANDIT_FILE.write_text(json.dumps(existing, indent=2))
        except Exception:
            pass

    # ── core logic ────────────────────────────────────────────────────────────

    def _arm_stats(self, regime: str, arm: str) -> list[int]:
        return self._stats.setdefault(regime, {}).setdefault(arm, [0, 0])

    def select(self, regime: str, recommended: str) -> str:
        """
        Thompson-sample all arms.  Returns recommended arm during warmup.
        Once any arm has >= WARMUP_TRIALS, sample all and return highest.
        Ensures recommended arm is in our arm list (safe fallback if not).
        """
        if recommended not in self.arms:
            return recommended

        total_trials = sum(sum(self._arm_stats(regime, a)) for a in self.arms)
        if total_trials < WARMUP_TRIALS:
            return recommended

        samples = {
            arm: random.betavariate(
                self._arm_stats(regime, arm)[0] + 1,
                self._arm_stats(regime, arm)[1] + 1,
            )
            for arm in self.arms
        }
        best = max(samples, key=lambda a: samples[a])

        return best

    def record(self, regime: str, arm: str, won: bool) -> None:
        if arm not in self.arms:
            return
        stats = self._arm_stats(regime, arm)
        stats[0 if won else 1] += 1
        self._save()

    # ── display ──────────────────────────────────────────────────────────────

    def arm_stats_for_regime(self, regime: str) -> dict:
        result = {}
        for arm in self.arms:
            w, l = self._arm_stats(regime, arm)
            total = w + l
            result[arm] = {
                "wins":    w,
                "losses":  l,
                "trials":  total,
                "winrate": round(w / total, 3) if total else None,
            }
        return result

    def summary(self) -> dict:
        out: dict = {}
        for regime in self._stats:
            out[regime] = self.arm_stats_for_regime(regime)
        return out


class DualStrategyBandit:
    """
    Wraps two bandits — one for passive mode, one for active mode.
    Provides the DualEngine with autonomous strategy selection
    that improves from every trade outcome.
    """

    # Evidence-based priors from backtesting and live bandit history.
    # Injected once at startup so the bandit starts informed, not naive.
    # Values = (synthetic_wins, synthetic_losses) per regime.
    # IMPORTANT: keep totals below WARMUP_TRIALS (8) so the bandit still
    # defers to regime recommendations during the true warmup period.
    # Evidence-based priors — intentionally small (1 observation per arm)
    # so the total per regime stays below WARMUP_TRIALS (8), preserving the
    # warmup deferral behaviour while still nudging the posterior correctly.
    # After 8 real trades the bandit takes over; priors become negligible.
    _ACTIVE_PRIORS: dict[str, dict[str, tuple[int, int]]] = {
        "sideways": {
            "funding_arb":    (1, 0),   # 100% wr prior
            "mean_reversion": (1, 0),   # 58.8% wr prior
            "momentum":       (0, 1),   # 9.2% wr — penalise
            "vwap_reversal":  (0, 1),   # 43.5% wr — slight penalty
            "breakout":       (1, 0),   # moderate prior
        },
        "bear": {
            "funding_arb":    (1, 0),
            "mean_reversion": (1, 0),
            "momentum":       (1, 0),
            "vwap_reversal":  (0, 1),
            "breakout":       (1, 0),
        },
        "bull": {
            "momentum":       (1, 0),
            "breakout":       (1, 0),
            "mean_reversion": (1, 0),
            "funding_arb":    (1, 0),
            "vwap_reversal":  (0, 1),
        },
    }

    def __init__(self):
        self.passive = ThompsonBandit(PASSIVE_ARMS, "passive")
        self.active  = ThompsonBandit(ACTIVE_ARMS,  "active")
        self._seed_priors()

    def _seed_priors(self) -> None:
        """
        Inject evidence-based priors once at startup.
        Only applied when the bandit has FEWER real trades than the synthetic count,
        so it never overwrites well-established real trade history.
        """
        for regime, modes in self._ACTIVE_PRIORS.items():
            for mode, (w, l) in modes.items():
                if mode not in ACTIVE_ARMS:
                    continue
                stats = self.active._arm_stats(regime, mode)
                real_trades = stats[0] + stats[1]
                synthetic   = w + l
                # Only seed if real history is sparse
                if real_trades < synthetic:
                    extra_w = max(0, w - stats[0])
                    extra_l = max(0, l - stats[1])
                    stats[0] += extra_w
                    stats[1] += extra_l
        self.active._save()

    # ── selection ─────────────────────────────────────────────────────────────

    def select_passive_mode(self, regime: str, recommended: str) -> str:
        return self.passive.select(regime, recommended)

    def select_active_mode(self, regime: str, recommended: str) -> str:
        return self.active.select(regime, recommended)

    # ── feedback ──────────────────────────────────────────────────────────────

    def record_passive(self, regime: str, mode: str, won: bool) -> None:
        self.passive.record(regime, mode, won)

    def record_active(self, regime: str, mode: str, won: bool) -> None:
        self.active.record(regime, mode, won)

    # ── snapshot for dashboard / state ────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "passive": self.passive.summary(),
            "active":  self.active.summary(),
        }

    def top_modes(self, regime: str) -> dict:
        """Return the current best mode per strategy for a given regime."""
        def _best(bandit: ThompsonBandit) -> str:
            stats = bandit.arm_stats_for_regime(regime)
            eligible = {
                arm: info["winrate"]
                for arm, info in stats.items()
                if info["trials"] >= WARMUP_TRIALS and info["winrate"] is not None
            }
            if not eligible:
                return "(warming up)"
            return max(eligible, key=lambda a: eligible[a])

        return {
            "passive": _best(self.passive),
            "active":  _best(self.active),
        }

    def apply_priority_hint(self, regime: str, priority: list[str]) -> None:
        """
        Inject AI Brain's mode priority as synthetic observations.

        Each position in the priority list adds a small prior: the first item
        gets +2 synthetic wins, the last item gets +0. This nudges Thompson
        Sampling toward the AI's recommendation without overriding real trade
        outcomes (which carry much larger weight after a few real trades).
        """
        boost = len(priority)  # e.g. 4 modes → boosts of 4, 3, 2, 1
        for mode in priority:
            if mode not in ACTIVE_ARMS:
                continue
            stats = self.active._arm_stats(regime, mode)
            synthetic_wins = max(0, boost - 1)
            stats[0] += synthetic_wins
            boost -= 1
        self.active._save()
