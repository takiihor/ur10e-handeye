import numpy as np
import pytest

from ur_handeye.collect import CollectOptions, Collector
from ur_handeye.dataset import Session
from ur_handeye.simulation import build_scenario
from ur_handeye.vision import BoardObserver


def run_auto(scenario, directory, **options):
    observer = BoardObserver(scenario.camera, scenario.target)
    session = Session(directory, {"robot": {"host": "simulated"}, "tcp_offset": scenario.robot.tcp_offset(),
                                  "board": scenario.target.describe(), "simulated": True})
    collector = Collector(scenario.robot, observer, session, CollectOptions(**options),
                          log=lambda *_: None, confirm=lambda summary: True)
    collector.run_auto()
    return collector, session


@pytest.fixture(scope="session")
def collected(tmp_path_factory):
    """One complete simulated calibration session, shared by the solve/CLI tests."""
    scenario = build_scenario(seed=3)
    directory = tmp_path_factory.mktemp("session") / "run"
    collector, _ = run_auto(scenario, directory)
    return scenario, directory, collector


@pytest.fixture
def rng():
    return np.random.default_rng(7)
