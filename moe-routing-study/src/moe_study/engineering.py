"""Run the task-harm and random-direction comparisons after one real update."""

import argparse
from functools import partial
import json
from pathlib import Path

import yaml

from moe_study.causal import run_cpu
from moe_study.config import RunConfig
from moe_study.engineering_measure import EngineeringMeasurement, write_engineering_report


def workload(protocol, sequences, layers, engines=2):
    direction = protocol["direction"]
    count = min(sequences, direction["sequences"])
    per_layer = direction["execution_repeats"] * (3 + 2 * direction["gaussian_pairs"])
    return {"optimizer_steps": 1,
        "full_dataset_forwards": engines * 5 * protocol["task_execution_repeats"],
        "task_sequences": sequences, "direction_sequences": count,
        "suffix_forwards_per_sequence_per_layer_per_engine": per_layer,
        "suffix_sequence_forwards_both_engines": engines * count * len(direction["layers"]) * per_layer,
        "suffix_transformer_layer_evaluations_both_engines": engines * count * per_layer * sum(layers - n for n in direction["layers"]),
        "note": "suffix calls skip the captured prefix; counts are work, not a wall-time or power promise"}


def run_engineering(config, args, protocol):
    metadata = {"definition": protocol,
        "workload": workload(protocol, args.sequences, protocol["model_layers"]),
        "task_order": "each old repeat N0, own-support sham; new even N1/sham/F1, odd F1/N1/sham",
        "direction_order": "seeded permutation of all suffix conditions per layer and execution, shared across ranks"}
    options = {"measurement_factory": partial(EngineeringMeasurement, protocol=protocol),
               "report_writer": partial(write_engineering_report, protocol=protocol),
               "full_dataset_forwards": 2 * 5 * protocol["task_execution_repeats"],
               "protocol_metadata": metadata}
    if config.machine["machine"]["backend"] == "cpu_development":
        run_cpu(config, args, **options)
    else:
        from moe_study.adapters.causal_train import run
        run(config, args, **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("machine", type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequences", type=int, default=1024)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    # Engineering capture uses the selected direction layers, with identical
    # hooks on every task condition. No full-length-only diagnostic subset.
    args.diagnostic_sequences, args.logit_block_size = 0, 32
    protocol = yaml.safe_load(args.protocol.read_text())
    config = RunConfig(yaml.safe_load(args.experiment.read_text()), yaml.safe_load(args.machine.read_text()),
                       {"segments": {"CAUSAL": {"resume": True, "final_analysis": False}}})
    if args.print_config:
        print(json.dumps({"config": config.expanded(), "layout": config.layout(), "protocol": protocol,
            "workload": workload(protocol, args.sequences, protocol["model_layers"]), "checkpoint": str(args.checkpoint),
            "output": str(args.output)}, indent=2))
        return
    run_engineering(config, args, protocol)


if __name__ == "__main__":
    main()
