import csv
import os
from datetime import datetime

FILE = "data/equity_curve.csv"

def log_equity(equity):

    os.makedirs("data", exist_ok=True)

    write_header = not os.path.exists(FILE)

    with open(FILE, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(["time","equity"])

        writer.writerow([datetime.utcnow(), equity])
