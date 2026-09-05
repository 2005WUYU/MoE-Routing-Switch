"""Three explicit configuration files: experiment, machine, schedule."""

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class RunConfig:
    experiment: dict
    machine: dict
    schedule: dict

    @classmethod
    def read(cls, experiment: str | Path, machine: str | Path, schedule: str | Path):
        return cls(*(yaml.safe_load(Path(path).read_text()) for path in (experiment, machine, schedule)))

    def expanded(self) -> dict:
        return {"experiment_file": self.experiment, "machine_file": self.machine, "schedule_file": self.schedule}

    def measurement_sequences(self, step: int) -> int:
        measurement = self.experiment["measurement"]
        if step in measurement["large_steps"]:
            return measurement["large_sequences"]
        if any(start <= step <= end for start, end in measurement["consecutive_windows"]):
            return measurement["window_sequences"]
        return 0

    def segment(self, name: str) -> dict:
        return self.schedule["segments"][name]

    def learning_rate(self, step: int) -> float:
        training = self.experiment["training"]
        return training["learning_rate"] * min(step / training["warmup_steps"], 1.0)

    def layout(self) -> dict:
        machine = self.machine["machine"]
        world = machine["nodes"] * machine["gpus_per_node"]
        dp = world // (machine["tensor_parallel"] * machine["pipeline_parallel"] * machine["context_parallel"])
        ep = machine["expert_parallel"]
        return {"world_size": world, "data_parallel": dp, "expert_data_parallel": dp // ep,
                "global_sequences": dp * machine["microbatch_sequences_per_gpu"] * machine["gradient_accumulation"],
                "expert_groups": [list(range(start, start + ep)) for start in range(0, world, ep)]}
