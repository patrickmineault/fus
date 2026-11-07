"""
Unit tests for the remote-zip helpers in lip_data.py (no network access needed).
"""

import struct
import zipfile

import numpy as np
import pytest

from lip_data import _timestamp_to_seconds, decompress_member, parse_zip_index


@pytest.fixture
def archive(tmp_path):
    """A small zip with a deflated and a stored member, plus its raw bytes."""
    rng = np.random.default_rng(0)
    contents = {
        "data/doppler/session.mat": rng.integers(0, 4, 200_000, dtype=np.uint8).tobytes(),
        "data/notes.json": b'{"Session": 99}',
    }
    path = tmp_path / "archive.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data/doppler/session.mat", contents["data/doppler/session.mat"], zipfile.ZIP_DEFLATED)
        zf.writestr("data/notes.json", contents["data/notes.json"], zipfile.ZIP_STORED)
    return path, path.read_bytes(), contents


def raw_member_bytes(blob, offset, compressed_size):
    """Compressed bytes of a member, skipping its local file header."""
    nlen, xlen = struct.unpack("<HH", blob[offset + 26 : offset + 30])
    start = offset + 30 + nlen + xlen
    return blob[start : start + compressed_size]


class TestParseZipIndex:
    def test_matches_zipfile(self, archive):
        path, blob, _ = archive
        members = parse_zip_index(blob[-65536:])
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                m = members[info.filename]
                assert m["offset"] == info.header_offset
                assert m["size"] == info.file_size
                assert m["compressed_size"] == info.compress_size
                assert m["crc"] == info.CRC
                assert m["method"] == info.compress_type

    def test_fetches_central_directory_outside_tail(self, archive):
        path, blob, _ = archive
        tail = blob[-22:]  # only the end-of-central-directory record
        fetched = []

        def fetch(start, end):
            fetched.append((start, end))
            return blob[start : end + 1]

        members = parse_zip_index(tail, fetch)
        assert len(fetched) == 1
        assert set(members) == {"data/doppler/session.mat", "data/notes.json"}

    def test_zip64(self, tmp_path):
        # More than 65535 entries forces zipfile to write ZIP64 end records
        path = tmp_path / "many.zip"
        n = 65_540
        with zipfile.ZipFile(path, "w") as zf:
            for i in range(n):
                zf.writestr(f"f{i}", b"")
            zf.writestr("last.bin", b"x" * 100)
        blob = path.read_bytes()
        assert blob.rfind(b"PK\x06\x06") >= 0

        members = parse_zip_index(blob[-65536:], lambda s, e: blob[s : e + 1])
        assert len(members) == n + 1
        with zipfile.ZipFile(path) as zf:
            assert members["last.bin"]["offset"] == zf.getinfo("last.bin").header_offset


class TestDecompressMember:
    @pytest.mark.parametrize("name", ["data/doppler/session.mat", "data/notes.json"])
    def test_roundtrip_and_crc(self, archive, tmp_path, name):
        _, blob, contents = archive
        m = parse_zip_index(blob[-65536:])[name]
        src = tmp_path / "member.part"
        src.write_bytes(raw_member_bytes(blob, m["offset"], m["compressed_size"]))

        dst = tmp_path / "member.out"
        crc, size = decompress_member(src, dst, m["method"])

        assert dst.read_bytes() == contents[name]
        assert (crc, size) == (m["crc"], m["size"])


def test_timestamp_to_seconds():
    # Behavioral timestamps are stored as HH:MM:SS,mmm
    assert _timestamp_to_seconds("08:35:12,864") == pytest.approx(8 * 3600 + 35 * 60 + 12.864)
