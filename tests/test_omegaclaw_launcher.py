import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "omega"


def _stub_docker_environment(tmp_path: Path) -> dict:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        "printf 'docker'\n"
        "printf ' <%s>' \"$@\"\n"
        "printf '\\n'\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)

    environment = os.environ.copy()
    environment["ASI_API_KEY"] = "test-token"
    environment["PATH"] = f"{bin_dir}{os.pathsep}{environment['PATH']}"
    return environment


def _run_launcher(tmp_path: Path, *component_options: str) -> subprocess.CompletedProcess:
    archive = tmp_path / "memory.tar.gz"
    archive.touch()

    environment = _stub_docker_environment(tmp_path)

    return subprocess.run(
        [
            str(LAUNCHER),
            "start",
            "--memory-transfer-dir",
            str(tmp_path),
            "--memory-import",
            archive.name,
            *component_options,
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("option", "included_environment", "excluded_environment"),
    [
        ("--only-history", "MEMORY_IMPORT_NO_VECTOR=1", "MEMORY_IMPORT_NO_HISTORY=1"),
        ("--only-vector", "MEMORY_IMPORT_NO_HISTORY=1", "MEMORY_IMPORT_NO_VECTOR=1"),
    ],
)
def test_only_component_options_select_one_import_component(
    tmp_path,
    option,
    included_environment,
    excluded_environment,
):
    result = _run_launcher(tmp_path, option)

    assert result.returncode == 0, result.stderr
    assert included_environment in result.stdout
    assert excluded_environment not in result.stdout


def test_only_component_options_are_mutually_exclusive(tmp_path):
    result = _run_launcher(tmp_path, "--only-history", "--only-vector")

    assert result.returncode != 0
    assert "--only-history and --only-vector cannot be combined" in result.stderr


@pytest.mark.parametrize("command", ["start", "stop", "clean"])
@pytest.mark.parametrize(
    ("flag", "expected_output"),
    [
        ("-h", "Usage:"),
        ("--help", "Usage:"),
        ("-v", "Omega version="),
        ("--version", "Omega version="),
    ],
)
def test_help_and_version_flags_do_not_run_the_command(tmp_path, command, flag, expected_output):
    result = subprocess.run(
        [str(LAUNCHER), command, flag],
        cwd=REPO_ROOT,
        env=_stub_docker_environment(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert expected_output in result.stdout
    assert "docker <" not in result.stdout


@pytest.mark.parametrize("removed_option", ["--no-history", "--no-vector"])
def test_removed_component_options_are_rejected(tmp_path, removed_option):
    result = _run_launcher(tmp_path, removed_option)

    assert result.returncode != 0
    assert "Usage:" in result.stdout
    assert "docker <" not in result.stdout
