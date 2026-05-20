"""
Strategy evolution — reallocates weight to what works, disables losers.
"""

from meta.self_reflection import SelfReflection


class EvolutionEngine:
    def __init__(self):
        self.reflection = SelfReflection()

    def evolve(self, strategies: list[dict]) -> list[dict]:
        evolved = []
        for s in strategies:
            sharpe = s.get("sharpe", 0)
            if sharpe < -0.5:
                s["allocation"] = 0
                s["enabled"] = False
                continue
            if sharpe < 0:
                s["allocation"] = max(s.get("allocation", 1) * 0.7, 0.1)
            elif sharpe > 1.5:
                s["allocation"] = min(s.get("allocation", 1) * 1.25, 2.0)
            elif sharpe > 2:
                s["allocation"] = min(s.get("allocation", 1) * 1.4, 2.5)
            s["enabled"] = s.get("allocation", 0) > 0
            evolved.append(s)
        return evolved

    def reflect_and_adjust(self, metrics: dict) -> dict:
        return self.reflection.evaluate(metrics)
