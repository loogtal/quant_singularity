#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv

load_dotenv()
from core.preflight import Preflight

if __name__ == "__main__":
    r = Preflight().run()
    Preflight().print_report(r)
    sys.exit(0 if r["ok"] else 1)
