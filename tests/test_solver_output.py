"""Tests for the pure-Python parts of solver output management.

The Julia wiring is exercised by the generation tests.
"""

import ctypes
import os

import pytest

from gridfm_datakit.process.solver_output import (
    LogChannel,
    LogRouter,
    SolverOutputConfig,
    SolverVerbosity,
    build_router,
    redirect_fds,
)


class TestSolverVerbosity:
    def test_parse_accepts_name_int_and_enum(self):
        assert SolverVerbosity.parse("debug") == SolverVerbosity.DEBUG
        assert SolverVerbosity.parse("  Silent ") == SolverVerbosity.SILENT
        assert SolverVerbosity.parse(2) == SolverVerbosity.WARNING
        assert SolverVerbosity.parse(SolverVerbosity.INFO) == SolverVerbosity.INFO

    def test_parse_rejects_unknown(self):
        with pytest.raises(ValueError):
            SolverVerbosity.parse("loud")


class TestSolverOutputConfig:
    def test_ipopt_and_memento_mapping(self):
        silent = SolverOutputConfig(SolverVerbosity.SILENT)
        assert silent.ipopt_print_level == 0
        # "not_set" means "inherit" in Memento (i.e. not silent); SILENT must map
        # to a level that actually suppresses everything.
        assert silent.memento_level == "emergency"

        debug = SolverOutputConfig(SolverVerbosity.DEBUG)
        assert debug.ipopt_print_level == 5
        assert debug.memento_level == "debug"

    def test_to_console_only_when_loud_and_no_log_dir(self):
        assert SolverOutputConfig(SolverVerbosity.INFO).to_console is True
        assert SolverOutputConfig(SolverVerbosity.SILENT).to_console is False
        assert (
            SolverOutputConfig(SolverVerbosity.INFO, log_dir="/tmp").to_console is False
        )

    def test_silence_powermodels_driven_by_level_only(self):
        # Silenced below INFO, regardless of console vs file destination -- so
        # DEBUG file logs genuinely contain PowerModels output.
        assert SolverOutputConfig(SolverVerbosity.SILENT).silence_powermodels is True
        assert SolverOutputConfig(SolverVerbosity.WARNING).silence_powermodels is True
        assert (
            SolverOutputConfig(
                SolverVerbosity.DEBUG,
                log_dir="/tmp",
            ).silence_powermodels
            is False
        )
        assert SolverOutputConfig(SolverVerbosity.INFO).silence_powermodels is False

    def test_from_settings_backcompat_defaults(self):
        # logs disabled -> silent, no dir
        cfg = SolverOutputConfig.from_settings(enable_solver_logs=False, log_dir="/x")
        assert cfg.verbosity == SolverVerbosity.SILENT
        assert cfg.log_dir is None

        # logs enabled -> debug, keep dir
        cfg = SolverOutputConfig.from_settings(enable_solver_logs=True, log_dir="/x")
        assert cfg.verbosity == SolverVerbosity.DEBUG
        assert cfg.log_dir == "/x"

        # explicit verbosity wins
        cfg = SolverOutputConfig.from_settings(
            verbosity="warning",
            enable_solver_logs=True,
            log_dir="/x",
        )
        assert cfg.verbosity == SolverVerbosity.WARNING


class TestRedirectCStdio:
    # Write straight to fd 1/2 (and via libc), not print(): the whole point is
    # capturing fd-level output that bypasses Python's sys.stdout.

    def test_redirects_c_level_writes_to_file(self, tmp_path, capfd):
        libc = ctypes.CDLL(None)
        log = tmp_path / "out.log"

        with redirect_fds(str(log)):
            os.write(1, b"fd-level stdout line\n")
            os.write(2, b"fd-level stderr line\n")
            libc.puts(b"c-level line")

        contents = log.read_text()
        assert "fd-level stdout line" in contents
        assert "fd-level stderr line" in contents
        assert "c-level line" in contents
        # Nothing leaked to the real stdout/stderr.
        captured = capfd.readouterr()
        assert captured.out == "" and captured.err == ""

    def test_discard_to_devnull(self, capfd):
        with redirect_fds(None):
            os.write(1, b"should be discarded\n")
        assert capfd.readouterr().out == ""

    def test_restores_fds_afterwards(self, tmp_path, capfd):
        log = tmp_path / "out.log"
        with redirect_fds(str(log)):
            os.write(1, b"inside\n")
        os.write(1, b"outside\n")
        # 'outside' reaches the real stdout again; 'inside' went to the file.
        assert capfd.readouterr().out.strip() == "outside"
        assert "inside" in log.read_text()

    def test_appends_rather_than_truncates(self, tmp_path):
        log = tmp_path / "out.log"
        log.write_text("preexisting\n")
        with redirect_fds(str(log)):
            os.write(1, b"added\n")
        text = log.read_text()
        assert "preexisting" in text and "added" in text

    def test_nested_capture_restores_correctly(self, tmp_path, capfd):
        inner = tmp_path / "inner.log"
        with redirect_fds(str(tmp_path / "outer.log")):
            with redirect_fds(str(inner)):
                os.write(1, b"deep\n")
            os.write(1, b"middle\n")
        os.write(1, b"surface\n")
        assert "deep" in inner.read_text()
        assert "middle" in (tmp_path / "outer.log").read_text()
        assert capfd.readouterr().out.strip() == "surface"


class TestLogChannel:
    def test_console_channel_passes_through(self, tmp_path, capfd):
        ch = LogChannel("opf", to_console=True)
        with ch.capture():
            os.write(1, b"visible\n")
        assert "visible" in capfd.readouterr().out

    def test_file_channel_captures_to_path(self, tmp_path, capfd):
        log = tmp_path / "opf.log"
        ch = LogChannel("opf", path=str(log))
        with ch.capture():
            os.write(1, b"to file\n")
        assert "to file" in log.read_text()
        assert capfd.readouterr().out == ""

    def test_discard_channel_swallows_output(self, capfd):
        ch = LogChannel("opf")  # no path, not console -> discard
        with ch.capture():
            os.write(1, b"gone\n")
        assert capfd.readouterr().out == ""

    def test_write_header_routes_with_capture(self, tmp_path, capfd):
        log = tmp_path / "opf.log"
        LogChannel("opf", path=str(log)).write_header("== warm ==")
        assert "== warm ==" in log.read_text()
        # Discard channel writes its header nowhere.
        LogChannel("opf").write_header("== warm ==")
        assert capfd.readouterr().out == ""


class TestLogRouter:
    def test_build_router_files_when_log_dir_set(self, tmp_path, capfd):
        cfg = SolverOutputConfig(SolverVerbosity.DEBUG, log_dir=str(tmp_path))
        router = build_router(cfg)
        with router.capture("opf"):
            os.write(1, b"opf output\n")
        # A file named opf_<process>.log holds it; nothing hit the console.
        opf_files = list(tmp_path.glob("opf_*.log"))
        assert opf_files and "opf output" in opf_files[0].read_text()
        assert capfd.readouterr().out == ""

    def test_build_router_discards_when_silent(self, capfd):
        router = build_router(SolverOutputConfig(SolverVerbosity.SILENT))
        with router.capture("pf"):
            os.write(1, b"noise\n")
        assert capfd.readouterr().out == ""

    def test_unknown_channel_discards(self, capfd):
        router = build_router(SolverOutputConfig(SolverVerbosity.INFO))
        # INFO with no log_dir -> known channels pass through to console, but an
        # unknown name defaults to discard rather than leaking.
        assert isinstance(router, LogRouter)
        with router.capture("mystery"):
            os.write(1, b"secret\n")
        assert capfd.readouterr().out == ""
