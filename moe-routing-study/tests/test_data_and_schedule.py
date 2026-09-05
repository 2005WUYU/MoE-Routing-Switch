from pathlib import Path

import numpy as np

from moe_study.config import RunConfig
from moe_study.data import TokenDataset, document_split, pack_document, prepare

ROOT = Path(__file__).resolve().parents[1]


def test_h20_work_count_and_dp_batch():
    config = RunConfig.read(ROOT / "experiments/h20_qwen_real_update.yaml", ROOT / "machines/h20_64.yaml",
                            ROOT / "schedules/h20_three_segments.yaml")
    counts = [config.measurement_sequences(step) for step in range(1, 513)]
    assert counts.count(1024) == 4
    assert counts.count(64) == 62
    assert sum(counts) * 2048 == 16515072
    assert config.layout()["data_parallel"] == 64
    assert config.layout()["expert_data_parallel"] == 8
    assert config.layout()["global_sequences"] == 128
    assert config.learning_rate(1) == 1e-5 / 32
    assert config.learning_rate(512) == 1e-5


def test_document_padding_shift_and_uint32():
    rows = list(pack_document([100000, 1, 2, 3, 4, 5], 4, 9, True))
    assert rows[0][0].dtype == np.dtype("<u4")
    assert rows[0][0].tolist() == [100000, 1, 2, 3, 4]
    assert rows[1][0].tolist() == [4, 5, 9, 9, 9]
    assert rows[1][1:] == (1, 4)
    assert len(list(pack_document([1, 2, 3, 4, 5, 6], 4, 9, False))) == 1


def test_prepare_keeps_document_pools_disjoint(tmp_path):
    documents = [{"id": str(i), "text": "1 2 3 4 5 6 7 8"} for i in range(100)]
    config = {"training": {"sequence_length": 4}, "data": {
        "train_tokens": 16, "measurement_sequences": 4, "measurement_split_percent": 50}}
    prepare(iter(documents), lambda text: list(map(int, text.split())), config, tmp_path, 0, 9, {"source": "test"})
    train, measure = TokenDataset(tmp_path, "train"), TokenDataset(tmp_path, "measurement")
    assert len(train) == len(measure) == 4
    assert set(row["document_id"] for row in train.index).isdisjoint(row["document_id"] for row in measure.index)
    assert all(row["valid_positions"] == 4 for row in train.index)
