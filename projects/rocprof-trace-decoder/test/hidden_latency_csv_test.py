#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from rocprof_trace_decoder import CodeIndex, Decoder, HiddenLatency, Pc, analyze_hidden_latency

ATT_RE = re.compile(r"_shader_engine_(\d+)_(\d+)\.att$", re.IGNORECASE)
HEADER = ["CodeObj", "Vaddr", "HiddenIdle", "HiddenStall", "HiddenIssue"]


def _expand(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        text = str(path)
        matches = sorted(glob.glob(text)) if any(char in text for char in "*?[]") else [text]
        if not matches:
            raise ValueError(f"No files matched: {path}")
        expanded.extend(Path(match).resolve() for match in matches)
    return expanded


def _shader_engine(path: Path) -> int:
    match = ATT_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot infer shader engine from ATT filename: {path}")
    return int(match.group(1))


def _read_expected(path: Path) -> dict[Pc, HiddenLatency]:
    expected: dict[Pc, HiddenLatency] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != HEADER:
            raise ValueError(f"Unexpected hidden-latency header in {path}: {reader.fieldnames}")
        for row in reader:
            pc = Pc(
                address=int(row["Vaddr"], 0),
                code_object_id=int(row["CodeObj"], 0),
            )
            expected[pc] = HiddenLatency(
                idle=int(row["HiddenIdle"], 0),
                stall=int(row["HiddenStall"], 0),
                issue=int(row["HiddenIssue"], 0),
            )
    return expected


def _add(target: dict[Pc, HiddenLatency], source: dict[Pc, HiddenLatency]) -> None:
    for pc, hidden in source.items():
        current = target.setdefault(pc, HiddenLatency())
        current += hidden


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate hidden-latency totals against a control CSV."
    )
    parser.add_argument("--lib", required=True, help="Path to the decoder shared library")
    parser.add_argument("--expected", required=True, type=Path, help="Hidden-latency CSV")
    parser.add_argument("att", nargs="+", type=Path, help="ATT trace files")
    parser.add_argument(
        "--stats",
        nargs="+",
        required=True,
        type=Path,
        help="Existing instruction-statistics control CSV files",
    )
    args = parser.parse_args()

    att_paths = _expand(args.att)
    stats_paths = _expand(args.stats)
    code_index = CodeIndex.from_stats_csv(stats_paths)
    actual: dict[Pc, HiddenLatency] = {}
    decoded_wave_count = 0
    with Decoder(args.lib) as decoder:
        for att_path in att_paths:
            records = decoder.parse_file(att_path, isa=code_index)
            decoded_wave_count += len(records.waves)
            result = analyze_hidden_latency(
                {_shader_engine(att_path): records},
                code_index=code_index,
            )
            _add(actual, result.by_pc)

    actual = {pc: value for pc, value in actual.items() if value.total()}
    if not decoded_wave_count:
        print("No wave records were decoded.")
        return 1

    expected = _read_expected(args.expected)
    if actual == expected:
        return 0

    for pc in sorted(actual.keys() | expected.keys()):
        if actual.get(pc) != expected.get(pc):
            print(
                f"PC {pc.code_object_id},{pc.address}: "
                f"actual={actual.get(pc)} expected={expected.get(pc)}"
            )
            break
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
