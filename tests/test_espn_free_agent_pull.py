"""The free-agent half of a league snapshot.

Rosters answer "who is taken"; the pool answers "who is available", which is
the only half that changes week to week. Two things are worth pinning: the
pool is pulled per position (ESPN ranks it globally, so an unfiltered call
buries whole positions), and a position that fails to fetch is recorded as a
failure rather than written out as an empty pool.
"""
import importlib.util
import json

import pytest

from config.settings import PROJECT_ROOT

SCRIPT = PROJECT_ROOT / "scripts" / "pull_espn_league.py"

STANDARD_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "RB/WR/TE": 1,
                  "D/ST": 1, "K": 1, "BE": 7, "IR": 2, "OP": 0, "P": 0}


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("pull_espn_league", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_standard_league_pulls_every_started_position(mod):
    assert mod._free_agent_positions({"position_slot_counts": STANDARD_SLOTS}) == [
        "QB", "RB", "WR", "TE", "K", "D/ST"]


def test_league_without_kickers_or_defenses_skips_them(mod):
    slots = dict(STANDARD_SLOTS, **{"K": 0, "D/ST": 0})
    assert mod._free_agent_positions({"position_slot_counts": slots}) == [
        "QB", "RB", "WR", "TE"]


def test_flex_slot_counts_for_each_component(mod):
    """A position startable only through the flex is still startable."""
    slots = {"QB": 1, "RB/WR/TE": 3, "BE": 6}
    assert mod._free_agent_positions({"position_slot_counts": slots}) == [
        "QB", "RB", "WR", "TE"]


def test_missing_settings_fall_back_to_the_standard_set(mod):
    """Unknown slots must not silently mean "pull nothing"."""
    assert mod._free_agent_positions({}) == list(mod.FREE_AGENT_POSITIONS)


class _StubConnector:
    """Enough of ESPNFantasyConnector to run one pull."""

    fail_positions = ()

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.requested = []

    def connect(self):
        return True

    def get_league_info(self):
        return {"name": "Test League", "num_teams": 2, "current_week": 1}

    def get_all_teams(self):
        return [{"team_id": 1, "team_name": "A"}, {"team_id": 2, "team_name": "B"}]

    def get_my_team(self, team_id=None, team_name=None):
        return {"team_id": team_id, "roster": [{"name": f"P{team_id}"}]}

    def get_league_settings(self):
        return {"position_slot_counts": STANDARD_SLOTS,
                "scoring_format": [{"abbr": "REC", "points": 1.0}]}

    def get_matchups(self):
        return [{"week": 1, "team_id": 1}]

    def get_free_agents(self, position=None, limit=25):
        self.requested.append((position, limit))
        if position in self.fail_positions:
            raise RuntimeError(f"boom {position}")
        return [{"name": f"{position}{i}", "position": position}
                for i in range(2)]


@pytest.fixture
def run_pull(mod, monkeypatch, tmp_path):
    """Run main() against a stub league, writing into tmp_path."""
    def run(fail_positions=(), argv=()):
        _StubConnector.fail_positions = fail_positions
        created = []

        def factory(**kwargs):
            c = _StubConnector(**kwargs)
            created.append(c)
            return c

        monkeypatch.setattr(mod, "ESPNFantasyConnector", factory)
        monkeypatch.setattr(mod, "ESPN_PRIVATE_DIR", tmp_path / "espn_private")
        monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
        monkeypatch.setenv("ESPN_LEAGUE_ID", "1")
        monkeypatch.setenv("ESPN_YEAR", "2026")
        monkeypatch.delenv("ESPN_S2", raising=False)
        monkeypatch.delenv("ESPN_SWID", raising=False)
        monkeypatch.setattr("sys.argv", ["pull_espn_league.py", *argv])

        assert mod.main() == 0
        snapshot = (tmp_path / "espn_private" / "latest").resolve()
        return created[0], snapshot
    return run


def test_snapshot_carries_the_pool_pulled_one_position_at_a_time(run_pull):
    connector, snapshot = run_pull()

    assert [pos for pos, _ in connector.requested] == [
        "QB", "RB", "WR", "TE", "K", "D/ST"]
    pool = json.loads((snapshot / "free_agents.json").read_text())
    assert len(pool) == 12
    manifest = json.loads((snapshot / "manifest.json").read_text())
    assert manifest["counts"]["free_agents"] == 12
    assert manifest["free_agents_by_position"]["QB"] == 2
    assert manifest["free_agent_failures"] == []


def test_fa_limit_reaches_the_connector(run_pull):
    connector, snapshot = run_pull(argv=["--fa-limit", "7"])

    assert {limit for _, limit in connector.requested} == {7}
    manifest = json.loads((snapshot / "manifest.json").read_text())
    assert manifest["free_agent_limit_per_position"] == 7


def test_one_failing_position_is_recorded_not_written_as_empty(run_pull):
    """The distinction the old swallow-and-return-[] made impossible."""
    _, snapshot = run_pull(fail_positions=("WR",))

    manifest = json.loads((snapshot / "manifest.json").read_text())
    assert manifest["free_agent_failures"] == [
        {"position": "WR", "error": "boom WR"}]
    assert "WR" not in manifest["free_agents_by_position"]
    # The rest of the snapshot still lands.
    assert manifest["counts"]["free_agents"] == 10
    assert manifest["counts"]["teams"] == 2
    assert (snapshot / "rosters.json").exists()
