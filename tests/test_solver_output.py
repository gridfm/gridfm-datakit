"""Tests for the pure-Python parts of solver output management.

The Julia wiring is exercised by the generation tests.
"""

import ctypes
import os

import pytest

from gridfm_datakit.process.solver_output import (
    SolverOutputConfig,
    SolverVerbosity,
    redirect_c_stdio,
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
        assert silent.memento_level == "not_set"

        debug = SolverOutputConfig(SolverVerbosity.DEBUG)
        assert debug.ipopt_print_level == 5
        assert debug.memento_level == "debug"

    def test_to_console_only_when_loud_and_no_log_dir(self):
        assert SolverOutputConfig(SolverVerbosity.INFO).to_console is True
        assert SolverOutputConfig(SolverVerbosity.SILENT).to_console is False
        assert (
            SolverOutputConfig(SolverVerbosity.INFO, log_dir="/tmp").to_console is False
        )

    def test_log_file_none_without_log_dir(self):
        assert SolverOutputConfig(SolverVerbosity.DEBUG).log_file("opf") is None

    def test_log_file_is_per_process(self, tmp_path):
        cfg = SolverOutputConfig(SolverVerbosity.DEBUG, log_dir=str(tmp_path))
        path = cfg.log_file("opf")
        assert path is not None
        assert path.startswith(str(tmp_path).replace("\\", "/"))
        assert os.path.basename(path).startswith("opf_")

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

        with redirect_c_stdio(str(log)):
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
        with redirect_c_stdio(None):
            os.write(1, b"should be discarded\n")
        assert capfd.readouterr().out == ""

    def test_restores_fds_afterwards(self, tmp_path, capfd):
        log = tmp_path / "out.log"
        with redirect_c_stdio(str(log)):
            os.write(1, b"inside\n")
        os.write(1, b"outside\n")
        # 'outside' reaches the real stdout again; 'inside' went to the file.
        assert capfd.readouterr().out.strip() == "outside"
        assert "inside" in log.read_text()

    def test_appends_rather_than_truncates(self, tmp_path):
        log = tmp_path / "out.log"
        log.write_text("preexisting\n")
        with redirect_c_stdio(str(log)):
            os.write(1, b"added\n")
        text = log.read_text()
        assert "preexisting" in text and "added" in text
