#!/usr/bin/env python3
"""Stage and run one TileCas optimizer across TileLang, CUDA C++, PTX and SASS."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "kernelopt/backends/cuda"
PROTOCOL = REPO / "kernelopt/protocol"
AGENTS = REPO / "kernelopt/agents"


def stage(args, workspace):
    with tempfile.TemporaryDirectory(dir=workspace.parent, prefix=".tilecas-") as tmp:
        run = Path(tmp) / "run"
        scripts = run / "scripts"
        candidates = run / "candidates"
        scripts.mkdir(parents=True)
        (run / "trajectory").mkdir()
        (run / ".harness").mkdir()
        (candidates / "initial").mkdir(parents=True)
        shutil.copyfile(args.reference, run / "reference.py")
        shutil.copyfile(args.tilelang, candidates / "initial/ModelNew.py")
        for asset in args.asset:
            if (run / asset.name).exists():
                raise ValueError(f"Asset name conflicts with a run file: {asset.name}")
            shutil.copyfile(asset, run / asset.name)
        shutil.copyfile(REPO / "skills/tilecas/SKILL.md", run / "TILECAS.md")
        for original, installed in (
            ("bench_layers.py", "bench.py"), ("lower_layers.py", "lower_layers.py"),
            ("diagnose_layers.py", "diagnose.py"), ("layer_runtime.py", "layer_runtime.py"),
            ("sass_edit.py", "sass_edit.py"), ("source_map.py", "source_map.py"),
        ):
            shutil.copyfile(BACKEND / original, scripts / installed)
        for name in ("module_loader.py", "validate_candidate.py"):
            shutil.copyfile(PROTOCOL / name, scripts / name)
        bench = scripts / "bench.sh"
        bench.write_text(
            '#!/usr/bin/env bash\n'
            'set -euo pipefail\n'
            'cd "$(dirname "$0")"\n'
            f'export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={args.gpu}\n'
            'exec python3 -u bench.py "$@"\n'
        )
        bench.chmod(0o755)
        (run / "settings.json").write_text(json.dumps({
            "gpu": args.gpu, "model": args.model, "effort": args.effort, "atol": args.atol,
            "rtol": args.rtol, "timing": "cuda_events", "layers": args.layers,
        }, indent=2) + "\n")
        (run / "ITERATIONS.md").write_text("# Iterations\n\n")
        (run / "AGENTS.md").write_text(
            "# CUDA TileCas\n\n"
            "Use one optimizer conversation and read TILECAS.md. Apply its search, evidence and "
            "stopping rules using the CUDA workspace layout and commands below.\n\n"
            f"- Editable layers in this run: {", ".join(args.layers)}.\n"
            "- Start by measuring the supplied TileLang implementation with "
            "`bash scripts/bench.sh tilelang initial baseline`.\n"
            "- Edit candidates under `candidates/`. Evaluate using "
            "`bash scripts/bench.sh LAYER CANDIDATE LABEL`. Each label is unique. "
            "The evaluation snapshots the source and measures all launches.\n"
            "- Run `python scripts/lower_layers.py MEASURED_LABEL CHILD` to materialize "
            "the next editable layer from a correct measured node. Add --to LAYER to skip intervening layers. "
            "TileLang lowers to CUDA C++, CUDA to PTX, and PTX to Hopper SASS edits. "
            "Branch or return to any retained candidate by copying its recorded source "
            "into a new candidate directory and setting its parent.txt to that label.\n"
            "- Edit CUDA device source and the corresponding launch ABI in program.json. "
            "Edit PTX directly. For SASS, consult kernel_N.sass and edit kernel_N.sass.json. "
            'Examples are `[{"op":"swap","offset":"0x100"}]` for adjacent instructions, '
            'and `[{"op":"stall","offset":"0x100","cycles":10}]` to increase a stall. '
            "The evaluator checks supported opcodes, encoding and dependencies.\n"
            "- When unsure about the next change, run `python scripts/diagnose.py`. "
            "Use the recorded parent, timing, CUDA source and machine code to choose a layer. "
            "For CUDA/PTX/SASS correspondence, use "
            "`python scripts/source_map.py LABEL --sass-offset OFFSET` or query a CUDA/PTX source line. "
            "Level changes and performance targets are optional.\n"
            "- Keep the input/output contract and use the same reference and evaluation for "
            "all candidates. Record each hypothesis and result in ITERATIONS.md, then "
            "verify the fastest retained candidate with the unique label final and explain "
            "the stopping decision. Trajectory snapshots and ITERATIONS.md are the candidate history "
            "for this workspace.\n"
            "- Use only the assigned GPU. Avoid editing evaluation files, reading other runs, "
            "or manually computing hashes or checksums.\n"
        )
        run.rename(workspace)


def launch(args, workspace, env):
    sys.path[:0] = [str(AGENTS), str(PROTOCOL)]
    import agent_driver
    import run_remote

    run_remote.ROOT = REPO
    run_remote.RUN_ROOT = workspace.parent
    run_remote.BACKEND = "cuda"
    run_remote.AKO_SKILL = REPO / "skills/tilecas/SKILL.md"
    agent_driver.EFFORT = args.effort
    cmd = agent_driver.build(
        args.model, "Read AGENTS.md and carry out this CUDA TileCas optimization.", workspace,
        single_agent=True, json_output=True,
    )
    cmd = run_remote.isolated_agent_command(
        workspace, cmd, workspace_view=workspace, private_history=True,
        readonly_workspace_paths=("reference.py", "settings.json", "scripts", "AGENTS.md", "TILECAS.md",
                                  *(asset.name for asset in args.asset)),
    )
    state = {"run_id": workspace.name, "task": args.reference.stem, "mode": "ours",
             "status": "running", "device": f"cuda:{args.gpu}", "agent_model": args.model,
             "reasoning_effort": args.effort, "started_at_epoch": time.time()}
    path = workspace / "state.json"
    run_remote.publish(path, state)
    status = "failed"
    try:
        with (workspace / "agent-events.jsonl").open("w") as log:
            result = subprocess.run(cmd, cwd=workspace, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=env)
        final = workspace / "trajectory/final/result.json"
        if result.returncode == 0 and final.is_file() and json.loads(final.read_text())["correct"]:
            status = "complete"
    finally:
        run_remote.publish(path, dict(state, status=status))
    return int(status != "complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="New run path, under the runs directory")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tilelang", type=Path, required=True, help="Initial ModelNew.py")
    parser.add_argument("--asset", type=Path, action="append", default=[])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model", default="gpt-6.2-sol")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--layers", nargs="+", choices=("tilelang", "cuda", "ptx", "sass"),
                        default=("tilelang", "cuda", "ptx", "sass"))
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if "tilelang" not in args.layers or len(set(args.layers)) != len(args.layers):
        parser.error("Include tilelang and list each layer once.")
    if workspace.exists():
        parser.error(f"Run already exists: {workspace}")
    if args.gpu < 0 or args.atol < 0 or args.rtol < 0:
        parser.error("GPU index and tolerances must be nonnegative")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    stage(args, workspace)
    print(workspace, flush=True)
    if args.stage_only:
        return 0
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(args.gpu))
    return launch(args, workspace, env)


if __name__ == "__main__":
    raise SystemExit(main())
