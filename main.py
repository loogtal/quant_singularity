"""Quant Singularity entry point."""

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    from core.dual_engine import DualEngine

    DualEngine().run()


if __name__ == "__main__":
    main()


