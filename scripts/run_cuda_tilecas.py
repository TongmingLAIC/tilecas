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
        initial = candidates / "initial"
        shutil.copyfile(args.reference, run / "reference.py")
        if args.initial_sass:
            locked = run / ".harness/sass-base"
            shutil.copytree(args.initial_sass, locked, ignore=shutil.ignore_patterns("parent.txt", "__pycache__"))
            shutil.copytree(locked, initial)
        else:
            initial.mkdir(parents=True)
            shutil.copyfile(args.tilelang, initial / "ModelNew.py")
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
        lowering = (
            "- Run `python scripts/lower_layers.py MEASURED_LABEL CHILD` to materialize "
            "the next editable layer from a correct measured node. Add --to LAYER to skip intervening layers. "
            "TileLang lowers to CUDA C++, CUDA to PTX, and PTX to Hopper SASS edits.\n"
        ) if args.mode == "ours" else (
            "- Change only kernel_N.sass.json edit plans. Keep the supplied cubins, input ABI, "
            "launch settings and PTX fixed. Inspect disassembly and profile the executable candidate. "
            "Apply the same evidence and stopping rules within SASS.\n"
        )
        (run / "AGENTS.md").write_text(
            f"# CUDA {'TileCas' if args.mode == 'ours' else 'SASS-only'}\n\n"
            "Use one optimizer conversation and read TILECAS.md. Apply its search, evidence and "
            "stopping rules using the CUDA workspace layout and commands below.\n\n"
            f"- Editable layers in this run: {", ".join(args.layers)}.\n"
            f"- Start by measuring the supplied implementation with "
            f"`bash scripts/bench.sh {args.initial_layer} initial baseline`.\n"
            "- Edit candidates under `candidates/`. Evaluate using "
            "`bash scripts/bench.sh LAYER CANDIDATE LABEL`. Each label is unique. "
            "The evaluation snapshots the source and measures all launches.\n"
            f"{lowering}"
            "- Branch or return to any retained candidate by copying its recorded source "
            "into a new candidate directory and setting its parent.txt to that label.\n"
            "- In editable CUDA or PTX layers, edit device source, PTX, or the launch ABI in program.json. "
            "For SASS, consult kernel_N.sass and edit kernel_N.sass.json. "
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
        args.model, f"Read AGENTS.md and optimize this CUDA kernel in {args.mode} mode.", workspace,
        single_agent=True, json_output=True,
    )
    readonly = ["reference.py", "settings.json", "scripts", "AGENTS.md", "TILECAS.md",
                *(asset.name for asset in args.asset)]
    if args.initial_sass:
        readonly.append(".harness/sass-base")
    cmd = run_remote.isolated_agent_command(
        workspace, cmd, workspace_view=workspace, private_history=True,
        readonly_workspace_paths=tuple(readonly),
    )
    state = {"run_id": workspace.name, "task": args.reference.stem, "mode": args.mode,
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
    initial = parser.add_mutually_exclusive_group(required=True)
    initial.add_argument("--tilelang", type=Path, help="Initial ModelNew.py")
    initial.add_argument("--initial-sass", type=Path, help="A candidate lowered to Hopper SASS")
    parser.add_argument("--asset", type=Path, action="append", default=[])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model", default="gpt-6.2-sol")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--layers", nargs="+", choices=("tilelang", "cuda", "ptx", "sass"))
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    args.mode = "sass-only" if args.initial_sass else "ours"
    args.initial_layer = "sass" if args.initial_sass else "tilelang"
    args.layers = args.layers or (["sass"] if args.initial_sass else ["tilelang", "cuda", "ptx", "sass"])
    if args.initial_sass:
        if args.layers != ["sass"]:
            parser.error("--initial-sass requires SASS-only editing")
        if not (args.initial_sass / "program.json").is_file():
            parser.error("--initial-sass requires a lowered candidate directory with a program.json")
    elif "tilelang" not in args.layers or len(set(args.layers)) != len(args.layers):
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
