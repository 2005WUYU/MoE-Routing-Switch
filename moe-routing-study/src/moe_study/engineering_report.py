"""Recompute the engineering report on Mac from compact document sums."""

import argparse
import json
from pathlib import Path

from moe_study.engineering_measure import write_engineering_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--engines", nargs="+", default=["bf16_execution", "fp32_reference"])
    args = parser.parse_args()
    protocol = json.loads((args.output / "protocol.json").read_text())
    write_engineering_report(args.output, args.engines, protocol=protocol)


if __name__ == "__main__":
    main()
