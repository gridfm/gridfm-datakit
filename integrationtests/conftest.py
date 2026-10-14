import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--calibrate",
        type=int,
        nargs="?",
        const=3,
        default=0,
        help="Calibrate metric bounds instead of asserting. Omit for normal assert mode "
        "(default 0). --calibrate alone runs 3 calibration passes; --calibrate N runs N. "
        "Example: pytest integrationtests --calibrate -s",
    )
    parser.addoption(
        "--ci",
        type=float,
        default=0.995,
        help="Confidence interval level for calibration stats (default 0.995). "
        "Example: pytest integrationtests --calibrate -s --ci 0.995",
    )
    parser.addoption(
        "--pad",
        type=float,
        default=0.001,
        help="Relative padding added to each calibrated bound as a floor on the "
        "margin of error (default 0.001 = 0.1%%). Seeded generation is bit-identical "
        "on one machine, so this floor is usually the whole interval; it absorbs "
        "solver jitter across machines. Metrics whose mean is 0 stay exactly (0, 0).",
    )


@pytest.fixture
def calibrate_runs(request):
    """Number of calibration runs via --calibrate (omitted = assert mode; flag alone = 3)."""
    return request.config.getoption("--calibrate")


@pytest.fixture
def ci_level(request):
    """Student-t confidence level for calibrated bounds via --ci (0.995 if omitted)."""
    return request.config.getoption("--ci")


@pytest.fixture
def calibrate_pad(request):
    """Relative half-width floor for calibrated bounds via --pad (0.001 if omitted)."""
    return request.config.getoption("--pad")
