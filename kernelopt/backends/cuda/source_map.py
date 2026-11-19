"""Look up CUDA/PTX lines and SASS offsets in a measured cubin."""

import argparse
import json
from pathlib import Path
import re
import subprocess

from layer_runtime import tool
from sass_edit import INSTRUCTION, SECTION


LINE = re.compile(r'^\s*//## File "([^"]+)", line (\d+)\b')
PTX_SOURCE = ".nv_debug_ptx_txt"


def parse(text, entry):
    selected = False
    cuda = ptx = None
    rows = {}
    for line in text.splitlines():
        if section := SECTION.match(line):
            selected = section[1] == entry
            cuda = ptx = None
        if not selected:
            continue
        if location := LINE.match(line):
            file, number = location[1], int(location[2])
            if file == PTX_SOURCE:
                ptx = number or None
            else:
                cuda = (file, number) if number else None
        if instruction := INSTRUCTION.match(line):
            rows[int(instruction[1], 16)] = {
                "instruction": instruction[2].strip(), "cuda": cuda, "ptx": ptx,
            }
    return rows


def measured_map(cubin, entry, edits):
    text = subprocess.check_output([tool("nvdisasm"), "-c", "-hex", "-g", "-gp", str(cubin)], text=True)
    rows = parse(text, entry)
    # The assembler's debug tables stay at their original offsets after a
    # size-preserving SASS swap. Move the source locations with the instructions.
    for edit in edits:
        if edit["op"] == "swap":
            first = int(edit["offset"], 0)
            second = first + 16
            for key in ("cuda", "ptx"):
                rows[first][key], rows[second][key] = rows[second][key], rows[first][key]
    return rows


def tilelang_source(trajectory, result, file):
    seen = set()
    parent = result.get("parent")
    while parent and parent not in seen and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", parent):
        seen.add(parent)
        node = trajectory / parent
        if not (node / "result.json").is_file():
            break
        result = json.loads((node / "result.json").read_text())
        if result["layer"] == "tilelang":
            return node / "source" / file
        parent = result.get("parent")
    return file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="Label of a measured CUDA, PTX or SASS candidate")
    parser.add_argument("--kernel", type=int, default=0)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--cuda-line", type=int)
    selector.add_argument("--ptx-line", type=int)
    selector.add_argument("--sass-offset", type=lambda value: int(value, 0))
    parser.add_argument("--cuda-file", help="Exact path or basename, default kernel_N.cu")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.label) or args.kernel < 0:
        parser.error("Use a trajectory label and a nonnegative kernel index")
    if args.cuda_file and args.cuda_line is None:
        parser.error("--cuda-file requires --cuda-line")
    root = Path(__file__).resolve().parents[1]
    node = root / "trajectory" / args.label
    source, artifacts = node / "source", node / "artifacts"
    if not (node / "result.json").is_file():
        parser.error("No result for this trajectory label")
    result = json.loads((node / "result.json").read_text())
    if not result["correct"] or result["layer"] == "tilelang":
        parser.error("Select a correct measured CUDA, PTX or SASS node")
    stem = f"kernel_{args.kernel}"
    kernels = json.loads((source / "program.json").read_text())["kernels"]
    if args.kernel >= len(kernels):
        parser.error("The measured candidate has no kernel at this index")
    spec = kernels[args.kernel]
    edits_path = source / f"{stem}.sass.json"
    edits = json.loads(edits_path.read_text()) if result["layer"] == "sass" else []
    rows = measured_map(artifacts / f"{stem}.cubin", spec["entry"], edits)
    source_file = args.cuda_file or f"{stem}.cu"
    callsite = spec.get("tilelang_callsite")
    print(f"{args.label} | {stem} | {spec['entry']}")
    if callsite:
        original = tilelang_source(root / "trajectory", result, callsite["file"])
        print(f"TileLang callsite: {original}:{callsite['line']}")
    print(f"Measured SASS: {artifacts / f'{stem}.sass'}")
    ptx_lines = (artifacts / f"{stem}.ptx").read_text().splitlines()
    found = False
    for offset, row in sorted(rows.items()):
        cuda, ptx = row["cuda"], row["ptx"]
        if args.sass_offset is not None and offset != args.sass_offset:
            continue
        if args.ptx_line is not None and ptx != args.ptx_line:
            continue
        if args.cuda_line is not None and (
            cuda is None or cuda[1] != args.cuda_line
            or source_file not in (cuda[0], Path(cuda[0]).name)
        ):
            continue
        found = True
        print(f"  {offset:#06x} {row['instruction']}")
        if cuda:
            recorded = Path(cuda[0])
            archived = source / "cuda-source" / recorded.name
            if not recorded.is_file() and recorded.name == f"{stem}.cu" and archived.is_file():
                recorded = archived
            print(f"    CUDA {recorded}:{cuda[1]}")
        if ptx and ptx <= len(ptx_lines):
            print(f"    PTX {stem}.ptx:{ptx}  {ptx_lines[ptx - 1].strip()}")
    if not found:
        print("No instruction with that source location or offset in the measured cubin.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
