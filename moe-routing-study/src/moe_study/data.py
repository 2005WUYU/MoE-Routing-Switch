"""Bounded streaming preparation and deterministic document-local sequence packing."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml


def document_split(document_id: str, measurement_percent: int) -> str:
    bucket = int.from_bytes(hashlib.sha256(document_id.encode()).digest()[:8], "big") % 100
    return "measurement" if bucket < measurement_percent else "train"


def pack_document(tokens: list[int], length: int, pad_id: int, padded: bool):
    """Every row stores length+1 IDs. Adjacent rows share their shift-boundary ID.

    Keeping each row within one source document gives unambiguous cluster IDs.
    Training drops the short document tail; measurement retains it with a mask.
    """
    for start in range(0, len(tokens) - 1, length):
        piece = tokens[start:start + length + 1]
        valid = len(piece) - 1
        if valid < length and not padded:
            break
        row = np.full(length + 1, pad_id, dtype="<u4")
        row[:len(piece)] = piece
        yield row, valid, start


class TokenDataset:
    def __init__(self, root: str | Path, split: str):
        root = Path(root)
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.length = self.manifest["sequence_length"]
        self.index = [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()]
        self.tokens = np.memmap(root / f"{split}.bin", dtype="<u4", mode="r",
                                shape=(len(self.index), self.length + 1))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        row = self.index[item]
        tokens = torch.from_numpy(self.tokens[item].astype(np.int64))
        return {"tokens": tokens[:-1], "labels": tokens[1:],
                "valid": torch.arange(self.length) < row["valid_positions"],
                "group": row["document_id"], "sequence": item}


def prepare(documents, tokenize, config: dict, destination: Path, pad_id: int, eos_id: int, provenance: dict):
    data, training = config["data"], config["training"]
    length = training["sequence_length"]
    targets = {"train": math.ceil(data["train_tokens"] / length),
               "measurement": data["measurement_sequences"]}
    counts = {"train": 0, "measurement": 0}
    destination.mkdir(parents=True, exist_ok=True)
    used_shards = dict.fromkeys([])
    from contextlib import ExitStack

    with ExitStack() as stack:
        binaries = {name: stack.enter_context((destination / f"{name}.bin").open("wb")) for name in targets}
        indices = {name: stack.enter_context((destination / f"{name}.jsonl").open("w")) for name in targets}
        for source_position, document in enumerate(documents):
            document_id = document["id"]
            split = document_split(document_id, data["measurement_split_percent"])
            if counts[split] == targets[split]:
                continue
            tokens = tokenize(document["text"]) + [eos_id]
            for row, valid, offset in pack_document(tokens, length, pad_id, split == "measurement"):
                if counts[split] == targets[split]:
                    break
                row.tofile(binaries[split])
                metadata = {"sequence": counts[split], "document_id": document_id,
                            "document_token_offset": offset, "valid_positions": valid,
                            "source_position": source_position, "source_shard": document.get("file_path")}
                indices[split].write(json.dumps(metadata, ensure_ascii=False) + "\n")
                used_shards[document.get("file_path")] = None
                counts[split] += 1
            if counts == targets:
                break
    manifest = {**provenance, "sequence_length": length, "storage_dtype": "uint32_little_endian",
                "packing": "document_local; stride=sequence_length; train drops short tails; measurement pads",
                "split": "sha256(document_id) first 8 bytes mod 100", "measurement_split_percent": data["measurement_split_percent"],
                "counts": counts, "requested_counts": targets, "used_shards_in_order": list(used_shards),
                "pad_id": pad_id, "eos_id": eos_id, "window_subset": "first_64"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.experiment.read_text())
    # Only this data-preparation entry imports download/tokenizer dependencies.
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    source = config["experiment"]["model_source"]
    data = config["data"]
    revision = HfApi().dataset_info(data["dataset"], revision=data["revision"]).sha
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    documents = load_dataset(data["dataset"], name=data["subset"], split="train", revision=revision, streaming=True)
    metadata = prepare(documents, lambda text: tokenizer.encode(text, add_special_tokens=False), config,
                       args.output or Path(data["root"]), tokenizer.eos_token_id, tokenizer.eos_token_id,
                       {"dataset": data["dataset"], "subset": data["subset"], "revision": revision,
                        "tokenizer_source": source})
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
