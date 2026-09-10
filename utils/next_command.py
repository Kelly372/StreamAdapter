"""Printable follow-up commands; never execute them automatically."""
import os
from pathlib import Path
import re
import shlex

from utils.project_paths import workspace_root
from utils.run_record import write_json


def portable_path(path, base):
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix()
    except ValueError:  # Different Windows drives cannot have a relative path.
        return Path(path).as_posix()


def next_run_name(folder, old, new):
    name = folder.name
    candidate = re.sub(rf"(^|_){re.escape(old)}(?=_|$)", rf"\g<1>{new}", name, count=1)
    if candidate == name:
        candidate = f"{name}_{new}"
    candidate = candidate[:160].rstrip(".")
    result, index = candidate, 2
    while (folder.parent / result).exists():
        result = f"{candidate}_{index}"
        index += 1
    return result


def data_context(config, repo):
    """Carry effective data/workspace settings across different stage configs."""
    root = Path(config.paths.workspace_root)
    args = []
    if root.resolve() != workspace_root(repo_root=repo):
        args += ["--workspace-root", portable_path(root, repo)]
    data = portable_path(config.paths.data_root, root)
    if data != "public_data/RealCam-Vid":
        args += ["--set", f"paths.data_root={data}"]
    return args


def print_next_command(folder, argv, repo):
    # Single quotes protect $, backticks and spaces in Bash and PowerShell.
    def quote(value):
        value = str(value)
        if re.fullmatch(r"[A-Za-z0-9_./:=,@%+\-]+", value):
            return value
        return "'" + value.replace("'", "''") + "'" if os.name == "nt" else shlex.quote(value)

    command = " ".join(quote(value) for value in argv)
    shell = "PowerShell" if os.name == "nt" else "POSIX shell"
    write_json(folder / "next_command.json", {"argv": argv, "cwd": str(repo), "shell": shell,
                                               "command": command})
    (folder / "next_command.txt").write_text(command + "\n", encoding="utf-8")
    print(f"Next command (run from the repository directory, {shell}):\n{command}", flush=True)
