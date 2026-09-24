"""The shared atomic write path.

Four copies of "temp file, then rename" were consolidated here; only one of
them fsynced, so the other three's "prevent corruption on crash" comment was
only half-true. These tests pin the properties that made consolidation worth
doing: an existing file is never clobbered by a failed write, no temp file
survives either outcome, and the JSON writer refuses to emit the bare NaN /
Infinity tokens that are not valid JSON.
"""
import json

import pandas as pd
import pytest

from src.utils.atomic_io import atomic_write, atomic_write_json, atomic_write_parquet


def test_atomic_write_creates_the_file_and_leaves_no_temp(tmp_path):
    target = tmp_path / "out.txt"
    atomic_write(target, lambda tmp: tmp.write_text("hello"))
    assert target.read_text() == "hello"
    assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]


def test_a_failed_write_leaves_the_previous_file_intact(tmp_path):
    """The whole point of writing through a temp file: a crash mid-write
    must not destroy the artifact that was already there.
    """
    target = tmp_path / "out.json"
    atomic_write_json({"generation": 1}, target)

    def _explode(tmp):
        tmp.write_text('{"generation": 2')  # truncated, then fail
        raise RuntimeError("writer blew up")

    with pytest.raises(RuntimeError, match="writer blew up"):
        atomic_write(target, _explode)

    assert json.loads(target.read_text()) == {"generation": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]


def test_a_writer_that_never_creates_the_file_still_cleans_up(tmp_path):
    target = tmp_path / "out.txt"
    with pytest.raises(FileNotFoundError):
        atomic_write(target, lambda tmp: tmp.unlink())
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_parent_directories_are_created(tmp_path):
    target = tmp_path / "a" / "b" / "out.json"
    atomic_write_json({"k": "v"}, target)
    assert json.loads(target.read_text()) == {"k": "v"}


def test_existing_file_is_replaced_in_place(tmp_path):
    target = tmp_path / "out.json"
    atomic_write_json({"n": 1}, target)
    atomic_write_json({"n": 2}, target)
    assert json.loads(target.read_text()) == {"n": 2}


def test_json_writer_rejects_nan_rather_than_writing_invalid_json(tmp_path):
    """Python's json defaults to emitting a bare `NaN` token, which no
    non-Python reader of these artifacts accepts. Fail at write time instead
    of persisting a file that only looks fine until the browser reads it.
    """
    target = tmp_path / "out.json"
    with pytest.raises(ValueError):
        atomic_write_json({"bad": float("nan")}, target)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_json_output_is_valid_and_newline_terminated(tmp_path):
    target = tmp_path / "out.json"
    atomic_write_json({"b": 2, "a": [1, 2, 3]}, target)
    text = target.read_text()
    assert text.endswith("\n")
    assert json.loads(text) == {"b": 2, "a": [1, 2, 3]}


def test_parquet_round_trips_without_the_index(tmp_path):
    target = tmp_path / "out.parquet"
    df = pd.DataFrame({"player_id": ["a", "b"], "points": [1.5, 2.5]})
    atomic_write_parquet(df, target)
    pd.testing.assert_frame_equal(pd.read_parquet(target), df)
    assert [p.name for p in tmp_path.iterdir()] == ["out.parquet"]


def test_temp_file_is_hidden_while_the_write_is_in_flight(tmp_path):
    """Several of these directories get globbed for real artifacts while a
    write may be running; a dotted temp name keeps it out of `*.parquet`.
    """
    target = tmp_path / "out.parquet"
    seen = {}

    def _writer(tmp):
        seen["name"] = tmp.name
        pd.DataFrame({"x": [1]}).to_parquet(tmp, index=False)

    atomic_write(target, _writer)
    assert seen["name"].startswith(".out.parquet.")
    assert list(tmp_path.glob("*.parquet")) == [target]
