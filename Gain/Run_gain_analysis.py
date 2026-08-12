#!/usr/bin/env python3
"""
Run the PMT gain analysis for selected folders/files.

This runner selects one configured lamp voltage per PMT serial, skips the
0.50 Vpp OFF files in the ON loop, and uses one OFF file per PMT serial.
The OFF file does not need to exist for every lamp voltage or HV setting.
"""

import os
import re
import subprocess
from pathlib import Path
import sys

BASE_PATH = Path(__file__).resolve().parent
ANALYSIS = BASE_PATH / "Gain_analysis_all.py"

PMT_FOLDERS = [
    "data_20260112",
    "data_20260126",
    "data_20260127",
    "data_20260218",
    "data_20260304",
    "data_20260305",
    "data_20260306",
]

BSL_LOW = 0
BSL_HIGH = 100
PK_LOW = 100
CHANNEL = 0

OFF_LAMP_VPP = 0.50
OFF_IDENTIFIERS = [
    "Lamp0.50_Vpp",
    "Lamp_0.50_Vpp",
    "Lamp0.50Vpp",
]

PYTHON = sys.executable

REQUIRE_TAG = None
# REQUIRE_TAG = "700Hz"
# REQUIRE_TAG = "-800V"

TARGET_LAMP_PER_PMT = {
    "LV2480": 1.90,
    "LV2481": 1.95,
    "LV2483": 1.95,
    "LV2485": 1.95,
}

LAMP_TOL = 1e-3


_PMT_RE = re.compile(r"(LV\d{4})")
_LAMP_RE = re.compile(r"Lamp_?([0-9]+(?:\.[0-9]+)?)_?Vpp", re.IGNORECASE)
_HV_RE = re.compile(r"(?<![A-Za-z0-9.])(-?\d{3,4})V(?!pp)", re.IGNORECASE)


def normalized_name(name):
    return name.replace("-", "_")


def extract_pmt_serial(filename):
    m = _PMT_RE.search(filename)
    return m.group(1) if m else None


def extract_lamp_vpp(filename):
    m = _LAMP_RE.search(normalized_name(filename))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def extract_hv(filename):
    matches = _HV_RE.findall(filename)
    if not matches:
        matches = _HV_RE.findall(normalized_name(filename))
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def has_required_tag(filename):
    return REQUIRE_TAG is None or REQUIRE_TAG in filename


def contains_off_identifier(filename):
    name = normalized_name(filename).lower()
    return any(normalized_name(tag).lower() in name for tag in OFF_IDENTIFIERS)


def is_off_file(filename):
    if contains_off_identifier(filename):
        return True
    lamp = extract_lamp_vpp(filename)
    return lamp is not None and abs(lamp - OFF_LAMP_VPP) <= LAMP_TOL


def is_on_candidate(filename):
    return (
        has_required_tag(filename)
        and "Lamp" in filename
        and not is_off_file(filename)
    )


def format_lamp(lamp):
    return "unknown" if lamp is None else f"{lamp:g} Vpp"


def format_hv(hv):
    return "unknown" if hv is None else f"{hv} V"


def find_off_file_for_pmt(files, pmt_serial):
    candidates = [
        f for f in files
        if has_required_tag(f)
        and pmt_serial in f
        and is_off_file(f)
    ]

    if not candidates:
        return None, "No OFF file with the same PMT serial was found."

    def sort_key(filename):
        lamp = extract_lamp_vpp(filename)
        lamp_distance = abs(lamp - OFF_LAMP_VPP) if lamp is not None else float("inf")
        hv = extract_hv(filename)
        hv_known_rank = 0 if hv is not None else 1
        return lamp_distance, hv_known_rank, filename

    chosen = sorted(candidates, key=sort_key)[0]
    if len(candidates) > 1:
        return chosen, (
            f"Multiple OFF files found for {pmt_serial}; using {chosen} for all selected ON files."
        )
    return chosen, None


def main():
    if not ANALYSIS.is_file():
        print(f"WARNING: analysis script does not exist yet: {ANALYSIS}")

    for pmt_folder in PMT_FOLDERS:
        folder = BASE_PATH / pmt_folder
        print(f"\nChecking folder: {folder}")

        if not folder.is_dir():
            print("Folder does not exist; skipping.")
            continue

        files = sorted(f for f in os.listdir(folder) if f.startswith("SPE"))
        if not files:
            print("No SPE files found; skipping.")
            continue

        off_file_by_pmt = {}

        for file_on in files:
            if not is_on_candidate(file_on):
                continue

            pmt_serial = extract_pmt_serial(file_on)
            if pmt_serial is None:
                print(f"Could not extract PMT serial from ON file: {file_on}")
                continue

            target_lamp = TARGET_LAMP_PER_PMT.get(pmt_serial)
            if target_lamp is None:
                print(f"No target lamp voltage configured for {pmt_serial}; skipping.")
                continue

            lamp_value = extract_lamp_vpp(file_on)
            if lamp_value is None:
                print(f"Could not extract lamp voltage from ON file: {file_on}")
                continue

            if abs(lamp_value - target_lamp) > LAMP_TOL:
                continue

            if pmt_serial not in off_file_by_pmt:
                file_off, note = find_off_file_for_pmt(files, pmt_serial)
                off_file_by_pmt[pmt_serial] = file_off
                if note:
                    print(f"WARNING for {pmt_serial}: {note}")
            else:
                file_off = off_file_by_pmt[pmt_serial]

            if file_off is None:
                continue

            hv_value = extract_hv(file_on)
            cmd = [
                PYTHON,
                str(ANALYSIS),
                "-pon", str(folder / file_on),
                "-poff", str(folder / file_off),
                "-bbl", str(BSL_LOW),
                "-bbu", str(BSL_HIGH),
                "-bpl", str(PK_LOW),
                "-c", str(CHANNEL),
            ]

            print("\n------------------------------------------------")
            print(f"Run analysis for PMT: {pmt_serial}")
            print(f"  Lamp voltage: {format_lamp(lamp_value)}")
            print(f"  HV: {format_hv(hv_value)}")
            print("------------------------------------------------")
            print(f"ON : {file_on}")
            print(f"OFF: {file_off}")

            try:
                result = subprocess.run(cmd, check=False)
            except FileNotFoundError as exc:
                print(f"ERROR: could not start analysis command: {exc}")
                continue

            if result.returncode != 0:
                print(f"WARNING: analysis exited with return code {result.returncode}")

    print("\nAll analyses finished.")


if __name__ == "__main__":
    main()