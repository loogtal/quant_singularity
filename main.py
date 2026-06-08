"""Quant Singularity entry point."""

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    from config.settings import _env_bool

    if _env_bool("QS_DUAL_MODE", False):
        from core.dual_engine import DualEngine

        DualEngine().run()
        return

    from core.engine import TradingEngine

    TradingEngine().run()


if __name__ == "__main__":
    main()


