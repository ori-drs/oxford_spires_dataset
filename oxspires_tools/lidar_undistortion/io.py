"""I/O utilities for PCD and IMU data."""

from pathlib import Path

import numpy as np
import pandas as pd


def _parse_pcd_header(f):
    """Parse PCD header fields, return dict and byte offset where data starts."""
    header = {}
    while True:
        line = f.readline().decode("ascii", errors="replace").strip()
        if line.startswith("DATA"):
            header["DATA"] = line.split()[1]
            break
        if line and not line.startswith("#"):
            key, *vals = line.split()
            header[key] = vals
    data_offset = f.tell()
    return header, data_offset


def _build_dtype(header):
    """Build a numpy dtype from PCD FIELDS/SIZE/TYPE/COUNT."""
    fields = header["FIELDS"]
    sizes = [int(s) for s in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(c) for c in header["COUNT"]]

    type_map = {"F": "f", "I": "i", "U": "u"}

    dt_list = []
    pad_idx = 0
    for name, size, typ, count in zip(fields, sizes, types, counts):
        np_kind = type_map[typ]
        base = f"{np_kind}{size}"
        if name == "_":
            total = size * count
            dt_list.append((f"_pad{pad_idx}", "u1", total))
            pad_idx += 1
        else:
            if count == 1:
                dt_list.append((name, base))
            else:
                dt_list.append((name, base, (count,)))
    return np.dtype(dt_list)


def read_pcd_binary(path: Path):
    """Read a binary PCD file. Returns structured numpy array and parsed header."""
    with open(path, "rb") as f:
        header, _ = _parse_pcd_header(f)
        n_points = int(header["POINTS"][0])
        dt = _build_dtype(header)
        data = np.frombuffer(f.read(n_points * dt.itemsize), dtype=dt)
    return data, header


def read_imu(path: Path) -> pd.DataFrame:
    """Load IMU CSV and add a unified timestamp_ns column."""
    df = pd.read_csv(path)
    df["timestamp_ns"] = df["secs"].astype(np.int64) * 10**9 + df["nsecs"].astype(np.int64)
    return df


def parse_scan_metadata(path: Path) -> dict:
    """Parse a scan metadata.txt file of key=value pairs."""
    meta = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()
    return meta


def save_pcd(path: Path, cloud: np.ndarray, header: dict, viewpoint: str = "0 0 0 1 0 0 0") -> None:
    """Save structured numpy array as binary PCD, preserving all fields from the original header."""
    n = len(cloud)
    fields = " ".join(header["FIELDS"])
    sizes = " ".join(header["SIZE"])
    types = " ".join(header["TYPE"])
    counts = " ".join(header["COUNT"])
    header_str = (
        f"VERSION 0.7\nFIELDS {fields}\nSIZE {sizes}\nTYPE {types}\nCOUNT {counts}\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT {viewpoint}\nPOINTS {n}\nDATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header_str.encode("ascii"))
        f.write(cloud.tobytes())
