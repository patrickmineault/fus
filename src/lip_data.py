"""
Download and load single sessions of the Griggs et al. (2025) LIP saccade dataset.

The dataset (https://data.caltech.edu/records/p5jan-02r60) is a single 56 GB zip
archive. Each recording session is a separate file inside it, so rather than
downloading everything we read the zip's table of contents and fetch only the
bytes of the session we want, using HTTP range requests through curl.
"""

import json
import os
import struct
import subprocess
import time
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

ARCHIVE_URL = "https://data.caltech.edu/records/p5jan-02r60/files/data_NC.zip?download=1"

# The probe has 128 elements with a 100 um pitch; the axial pixel size is stored in the file
PIXEL_X_MM = 0.1

EVENTS = (
    "trialstart",
    "fixationhold",
    "cue",
    "memory",
    "target_acquire",
    "target_hold",
    "reward",
    "iti",
)

# ========== Remote zip access ==========


def _curl(args, out_path):
    return subprocess.run(
        ["curl", "-sSL", "--fail", "--max-time", "600", *args, "-o", str(out_path)],
        capture_output=True,
    )


def _fetch_range(start, end, tmp_path, url=ARCHIVE_URL, retries=8):
    """Return bytes [start, end] (inclusive) of `url`, retrying flaky connections."""
    for attempt in range(retries):
        result = _curl(["-r", f"{start}-{end}", url], tmp_path)
        if result.returncode == 0 and os.path.getsize(tmp_path) == end - start + 1:
            data = Path(tmp_path).read_bytes()
            os.remove(tmp_path)
            return data
        time.sleep(min(2 ** (attempt + 1), 30))  # back off before retrying
    raise IOError(f"Could not download bytes {start}-{end}: {result.stderr.decode()}")


def list_archive(dest_dir, url=ARCHIVE_URL):
    """
    Read the table of contents (the "central directory") of the remote zip.

    Returns
    -------
    members : dict
        file name -> dict(offset, compressed_size, size, method, crc)
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    cache = dest_dir / "archive_index.json"
    if cache.exists():
        return json.loads(cache.read_text())

    # The central directory sits at the very end of a zip file. A suffix range
    # ("-65536") asks for the last 64 kB without needing to know the file size.
    tmp = dest_dir / "tail.part"
    for _ in range(8):
        result = _curl(["-r", "-65536", url], tmp)
        if result.returncode == 0:
            break
    else:
        raise IOError(result.stderr.decode())
    tail = tmp.read_bytes()
    os.remove(tmp)

    members = parse_zip_index(
        tail, lambda start, end: _fetch_range(start, end, dest_dir / "cd.part", url)
    )
    cache.write_text(json.dumps(members))
    return members


def parse_zip_index(tail, fetch=None):
    """
    Parse a zip's central directory, given the last bytes of the archive.

    Parameters
    ----------
    tail : bytes
        The end of the archive, containing at least the end-of-central-directory records
    fetch : callable or None
        fetch(start, end) -> bytes, used if the central directory isn't contained in `tail`

    Returns
    -------
    members : dict
        file name -> dict(offset, compressed_size, size, method, crc)
    """
    # Archives > 4 GB store sizes and offsets in a ZIP64 record; small ones in the classic record
    i = tail.rfind(b"PK\x06\x06")
    if i >= 0:
        n_entries, cd_size, cd_offset = struct.unpack("<QQQ", tail[i + 32 : i + 56])
    else:
        i = tail.rfind(b"PK\x05\x06")
        assert i >= 0, "End-of-central-directory record not found"
        n_entries, cd_size, cd_offset = struct.unpack("<HII", tail[i + 10 : i + 20])

    # The central directory immediately precedes the end records
    if i >= cd_size and tail[i - cd_size : i - cd_size + 4] == b"PK\x01\x02":
        cd = tail[i - cd_size : i]
    else:
        cd = fetch(cd_offset, cd_offset + cd_size - 1)

    members, p = {}, 0
    while p + 46 <= len(cd) and cd[p : p + 4] == b"PK\x01\x02":
        method, _, _, crc, csize, usize, nlen, xlen, clen = struct.unpack(
            "<HHHIIIHHH", cd[p + 10 : p + 34]
        )
        offset = struct.unpack("<I", cd[p + 42 : p + 46])[0]
        name = cd[p + 46 : p + 46 + nlen].decode("utf-8")
        extra = cd[p + 46 + nlen : p + 46 + nlen + xlen]

        # Values that overflow 32 bits are stored in the ZIP64 "extra" field instead
        q = 0
        while q + 4 <= len(extra):
            tag, size = struct.unpack("<HH", extra[q : q + 4])
            if tag == 0x0001:
                vals = list(struct.unpack(f"<{size // 8}Q", extra[q + 4 : q + 4 + size]))
                if usize == 0xFFFFFFFF:
                    usize = vals.pop(0)
                if csize == 0xFFFFFFFF:
                    csize = vals.pop(0)
                if offset == 0xFFFFFFFF:
                    offset = vals.pop(0)
            q += 4 + size

        members[name] = dict(
            offset=offset, compressed_size=csize, size=usize, method=method, crc=crc
        )
        p += 46 + nlen + xlen + clen

    assert len(members) == n_entries, "Incomplete central directory"
    return members


def fetch_member(name, dest_dir, url=ARCHIVE_URL, chunk_mb=16):
    """
    Download a single file from the remote zip into `dest_dir`.

    The compressed bytes are fetched in chunks (resuming if interrupted),
    then decompressed and checked against the CRC-32 stored in the archive.
    """
    dest_dir = Path(dest_dir)
    out = dest_dir / Path(name).name
    m = list_archive(dest_dir, url)[name]
    if out.exists() and out.stat().st_size == m["size"]:
        return out

    # Each member starts with a 30-byte local header followed by its name and extra field
    header = _fetch_range(m["offset"], m["offset"] + 29, dest_dir / "header.part", url)
    nlen, xlen = struct.unpack("<HH", header[26:30])
    data_start = m["offset"] + 30 + nlen + xlen

    part = dest_dir / (out.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    chunk = chunk_mb * 2**20
    with open(part, "ab") as fh, tqdm(
        total=m["compressed_size"], initial=done, unit="B", unit_scale=True, desc=out.name
    ) as bar:
        while done < m["compressed_size"]:
            end = min(done + chunk, m["compressed_size"]) - 1
            data = _fetch_range(data_start + done, data_start + end, dest_dir / "chunk.part", url)
            fh.write(data)
            done += len(data)
            bar.update(len(data))

    crc, size = decompress_member(part, out, m["method"])
    if crc != m["crc"] or size != m["size"]:
        os.remove(out)
        os.remove(part)
        raise IOError(f"{name}: checksum mismatch, please re-run to download again")
    os.remove(part)
    return out


def decompress_member(src_path, dst_path, method):
    """
    Decompress the raw bytes of one zip member (stored or deflated).

    Returns the CRC-32 and size of the decompressed data, to check against the archive.
    """
    crc, size = 0, 0
    # zip "deflate" is a raw zlib stream, without zlib's own header
    inflater = zlib.decompressobj(-15) if method == 8 else None
    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        while block := src.read(2**22):
            block = inflater.decompress(block) if inflater else block
            crc, size = zlib.crc32(block, crc), size + len(block)
            dst.write(block)
        if inflater:
            block = inflater.flush()
            crc, size = zlib.crc32(block, crc), size + len(block)
            dst.write(block)
    return crc, size


def fetch_session(session, run, dest_dir, url=ARCHIVE_URL):
    """Download the trial-aligned Power Doppler file of one session."""
    return fetch_member(f"data/doppler/doppler_S{session}_R{run}+normcorre.mat", dest_dir, url)


def session_table(dest_dir, url=ARCHIVE_URL):
    """List all recording sessions (monkey, imaging plane, number of trials...)."""
    path = fetch_member("data/ProjectRecord_paper.json", dest_dir, url)
    return json.loads(Path(path).read_text())


# ========== Loading a session ==========


@dataclass
class Session:
    """One recording session, with trials aligned to trial start."""

    doppler: np.ndarray  # (n_trials, n_frames, nz, nx) Power Doppler, 1 frame / second
    angiogram: np.ndarray  # (nz, nx) mean Power Doppler image
    target_deg: np.ndarray  # (n_trials,) saccade direction, 0 = right, 90 = up
    events: dict  # event name -> (n_trials,) time in seconds after trial start
    frame_rate: float  # Doppler frames per second
    dx_mm: float  # lateral pixel size
    dz_mm: float  # axial pixel size
    depth_mm: tuple  # (top, bottom) depth of the image below the probe

    @property
    def extent(self):
        """imshow extent in mm: lateral position, depth below the probe."""
        nz, nx = self.angiogram.shape
        return [0, nx * self.dx_mm, self.depth_mm[0] + nz * self.dz_mm, self.depth_mm[0]]


def _timestamp_to_seconds(t):
    t = datetime.strptime(t, "%H:%M:%S,%f")
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def load_session(path):
    """Load a trial-aligned MATLAB v7.3 (HDF5) session file."""
    with h5py.File(path, "r") as f:

        def deref(ref):
            obj = f[ref]
            if obj.attrs.get("MATLAB_class") == b"char":
                return "".join(map(chr, obj[()].ravel()))
            return np.squeeze(obj[()])

        behavior = f["behavior"]
        target_pos = np.array([deref(r) for r in behavior["targetPos"][:, 0]])
        trial_start = np.array(
            [_timestamp_to_seconds(deref(r)) for r in behavior["trialstart"][:, 0]]
        )
        events = {
            ev: np.array([_timestamp_to_seconds(deref(r)) for r in behavior[ev][:, 0]])
            - trial_start
            for ev in EVENTS
        }

        # Older files store the Doppler frame rate; newer ones only the acquisition loop period
        if "framerate" in f["coreParams"]:
            frame_rate = float(np.squeeze(f["coreParams"]["framerate"][()]))
        else:
            frame_rate = 1 / float(np.squeeze(f["UF"]["Time_loop"][()]))

        # MATLAB stores arrays column-major, so h5py sees the axes reversed:
        # (trial, frame, x, z) -> we reorder to (trial, frame, z, x)
        return Session(
            doppler=f["iDop"][()].transpose(0, 1, 3, 2),
            angiogram=f["angiogram"][()].T,
            target_deg=np.round(np.degrees(np.arctan2(target_pos[:, 1], target_pos[:, 0]))) % 360,
            events=events,
            frame_rate=frame_rate,
            dx_mm=PIXEL_X_MM,
            dz_mm=float(np.squeeze(f["UF"]["Lambda"][()])),
            depth_mm=tuple(np.squeeze(f["UF"]["Depth"][()])),
        )
