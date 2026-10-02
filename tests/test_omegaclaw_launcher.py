import os
import shutil
import shlex
import stat
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
from scripts import memory_transfer


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "omega"
CONTAINER_TEST_IMAGE = os.environ.get("OMEGA_LAUNCHER_TEST_IMAGE", "")


def _load_installer_namespace():
    launcher_source = LAUNCHER.read_text(encoding="utf-8")
    installer_source = launcher_source.split("cat >\"$tmp_py_file\" <<'PY'\n", 1)[1].split(
        "\nPY\n", 1
    )[0]
    namespace = {"__name__": "omega_installer"}
    previous_script_dir = os.environ.get("OMEGA_SCRIPT_DIR")
    os.environ["OMEGA_SCRIPT_DIR"] = str(REPO_ROOT / "scripts")
    try:
        exec(compile(installer_source, str(LAUNCHER), "exec"), namespace)
    finally:
        if previous_script_dir is None:
            del os.environ["OMEGA_SCRIPT_DIR"]
        else:
            os.environ["OMEGA_SCRIPT_DIR"] = previous_script_dir
    return namespace


def _run_launcher(
    tmp_path: Path,
    *component_options: str,
    transfer_dir: Path | None = None,
    transfer_gid: int | None = None,
    memory_import: bool = True,
) -> subprocess.CompletedProcess:
    if transfer_dir is None:
        transfer_dir = tmp_path
        transfer_dir.chmod(0o2770)
    transfer_gid = transfer_gid if transfer_gid is not None else transfer_dir.stat().st_gid
    launcher_options = [
        "--memory-transfer-dir",
        str(transfer_dir),
        "--memory-transfer-gid",
        str(transfer_gid),
    ]
    if memory_import:
        archive = transfer_dir / "memory.tar.gz"
        archive.touch()
        launcher_options.extend(["--memory-import", archive.name])

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
    uname = bin_dir / "uname"
    uname.write_text("#!/bin/sh\necho Linux\n", encoding="utf-8")
    uname.chmod(0o755)
    python3 = bin_dir / "python3"
    python3.write_text(
        "#!/bin/sh\n"
        "if [ \"${1##*/}\" = \"memory_transfer.py\" ]; then\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = \"-c\" ]; then\n"
        f"  echo '{os.geteuid()} {os.geteuid()} {transfer_gid} 1528'\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$1\" = \"-\" ]; then\n"
        "  if grep -qx '    else:'; then exit 1; fi\n"
        "  exit 2\n"
        "fi\n"
        f"exec {shlex.quote(sys.executable)} \"$@\"\n",
        encoding="utf-8",
    )
    python3.chmod(0o755)

    environment = os.environ.copy()
    environment["ASI_API_KEY"] = "test-token"
    environment["PATH"] = f"{bin_dir}{os.pathsep}{environment['PATH']}"

    return subprocess.run(
        [
            str(LAUNCHER),
            "start",
            *launcher_options,
            *component_options,
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_installer_records_private_group_for_existing_transfer_directory(tmp_path, monkeypatch):
    transfer_dir = tmp_path / "memory-transfer"
    transfer_dir.mkdir()
    group_id = transfer_dir.stat().st_gid
    answers = iter(["y", str(transfer_dir), str(group_id), "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    installer = _load_installer_namespace()
    monkeypatch.setitem(installer, "validate_memory_transfer_dir", lambda *_: None)
    memory_transfer_dir, memory_transfer_gid, memory_export_enabled = installer["_choose_memory_transfer"]()

    assert memory_transfer_dir == str(transfer_dir)
    assert memory_transfer_gid == str(group_id)
    assert memory_export_enabled == "0"


def test_installer_rejects_unprepared_transfer_directory_before_export_prompt(tmp_path, monkeypatch):
    transfer_dir = tmp_path / "memory-transfer"
    transfer_dir.mkdir()
    group_id = transfer_dir.stat().st_gid
    answers = iter(["y", str(transfer_dir), str(group_id)])
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or next(answers))
    installer = _load_installer_namespace()
    monkeypatch.setitem(
        installer,
        "validate_memory_transfer_dir",
        lambda *_: "--memory-transfer-dir must enable setgid",
    )

    with pytest.raises(StopIteration):
        installer["_choose_memory_transfer"]()

    assert "Enable memory export for this instance? [y/N]: " not in prompts


def test_memory_transfer_validator_accepts_group_protected_directory(tmp_path, monkeypatch):
    transfer_dir = tmp_path / "memory-transfer"
    transfer_dir.mkdir()
    group_id = transfer_dir.stat().st_gid
    monkeypatch.setattr(memory_transfer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        memory_transfer.os,
        "stat",
        lambda _: SimpleNamespace(
            st_uid=os.geteuid(),
            st_gid=group_id,
            st_mode=stat.S_IFDIR | stat.S_ISGID | 0o770,
        ),
    )
    monkeypatch.setattr(memory_transfer, "_has_posix_acl", lambda _: False)

    assert memory_transfer.validate_memory_transfer_dir(
        str(transfer_dir), str(group_id)
    ) is None


def test_launcher_uses_private_group_for_preflight_and_entrypoint(tmp_path):
    transfer_dir = tmp_path / "memory-transfer"
    transfer_dir.mkdir()
    transfer_dir.chmod(0o2770)

    result = _run_launcher(tmp_path, transfer_dir=transfer_dir, memory_import=False)

    assert result.returncode == 0, result.stderr
    group_id = transfer_dir.stat().st_gid
    assert f"<--user> <65534:65534> <--group-add> <{group_id}>" in result.stdout
    assert f"<-e> <MEMORY_TRANSFER_GID={group_id}>" in result.stdout


def test_entrypoint_resolves_transfer_gid_to_a_supplementary_group_name():
    entrypoint = (REPO_ROOT / "entrypoint.sh").read_text(encoding="utf-8")

    assert 'getent group "${MEMORY_TRANSFER_GID}"' in entrypoint
    assert 'groupadd --gid "${MEMORY_TRANSFER_GID}" "${memory_transfer_group}"' in entrypoint
    assert 'su --group nogroup --supp-group "${memory_transfer_group}" nobody' in entrypoint


@pytest.mark.skipif(
    sys.platform != "linux" or not CONTAINER_TEST_IMAGE or shutil.which("docker") is None,
    reason="needs Linux, Docker, and OMEGA_LAUNCHER_TEST_IMAGE",
)
def test_container_user_can_write_group_protected_transfer_directory(tmp_path):
    transfer_dir = tmp_path / "memory-transfer"
    transfer_dir.mkdir()
    transfer_dir.chmod(0o2770)

    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "65534:65534",
            "--group-add",
            str(transfer_dir.stat().st_gid),
            "--entrypoint",
            "/bin/sh",
            "--volume",
            f"{transfer_dir}:/memory-transfer",
            CONTAINER_TEST_IMAGE,
            "-c",
            (
                "touch /memory-transfer/probe && "
                f"test \"$(stat -c %g /memory-transfer/probe)\" = \"{transfer_dir.stat().st_gid}\" && "
                "rm /memory-transfer/probe"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


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


@pytest.mark.parametrize("removed_option", ["--no-history", "--no-vector"])
def test_removed_component_options_are_rejected(tmp_path, removed_option):
    result = _run_launcher(tmp_path, removed_option)

    assert result.returncode != 0
    assert "Usage:" in result.stdout
    assert "docker <" not in result.stdout
