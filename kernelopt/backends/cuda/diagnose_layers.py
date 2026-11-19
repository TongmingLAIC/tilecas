"""Report the retained candidates and profile their measured launch graphs."""

from collections import defaultdict
import json
import os
from pathlib import Path
import sys


def profile(root, row, node, settings):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(settings["gpu"])

    import torch
    from torch.profiler import ProfilerActivity, profile
    from layer_runtime import Runtime
    from lower_layers import make_model
    from module_loader import load

    reference = load(root / "reference.py", "tilecas_reference")
    source, artifacts = node / "source", node / "artifacts"
    program, runtime = None, None
    if row["layer"] == "tilelang":
        candidate = make_model(source / "ModelNew.py", reference)
    else:
        program = json.loads((source / "program.json").read_text())
        runtime = Runtime(program, artifacts)
        candidate = runtime

    try:
        torch.manual_seed(42)
        repeats = 5
        with torch.inference_mode():
            inputs = reference.get_inputs()
            for _ in range(repeats):
                candidate(*inputs)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
                for _ in range(repeats):
                    candidate(*inputs)
                    torch.cuda.synchronize()
        activities = sorted(
            (event for event in trace.events()
             if event.device_type == torch.autograd.DeviceType.CUDA),
            key=lambda event: event.time_range.start,
        )
        if not activities:
            print("The CUDA profiler reported no device activities.")
            return
        print(f"CUDA activities for {row['label']} (mean microseconds per candidate call)")
        launches = launch_timings(activities, repeats, program)
        if launches is None:
            durations = defaultdict(float)
            for event in activities:
                durations[event.name] += event.time_range.elapsed_us() / repeats
            print("The activity sequence could not be matched to individual launches.")
            for name, elapsed in sorted(durations.items(), key=lambda item: -item[1]):
                print(f"  {elapsed:.3f} us | {name}")
            return
        for elapsed, index, name in sorted(launches, key=lambda launch: -launch[0]):
            if program is None:
                print(f"  {elapsed:.3f} us | launch {index} | {name}")
                continue
            spec = program["kernels"][index]
            callsite = spec.get("tilelang_callsite")
            origin = f" | TileLang {callsite['file']}:{callsite['line']}" if callsite else ""
            print(f"  {elapsed:.3f} us | kernel_{index}: {name} | grid {spec['grid']}"
                  f" | block {spec['block']} | {spec['shared_bytes']} dynamic shared bytes{origin}")
            sass = artifacts / f"kernel_{index}.sass"
            print(f"    SASS {sass.relative_to(root)}")
    finally:
        if runtime is not None:
            runtime.close()


def launch_timings(activities, repeats, program):
    count, remainder = divmod(len(activities), repeats)
    if remainder or not count:
        return None
    if program is None:
        names = [event.name for event in activities[:count]]
    else:
        names = [spec["entry"] for spec in program["kernels"]]
    if [event.name for event in activities] != names * repeats:
        return None
    return [
        (sum(event.time_range.elapsed_us() for event in activities[index::count]) / repeats,
         index, name)
        for index, name in enumerate(names)
    ]


def main():
    root = Path(__file__).resolve().parents[1]
    records = []
    for path in sorted(root.glob("trajectory/*/result.json")):
        row = json.loads(path.read_text())
        if row["correct"]:
            records.append((row, path.parent))
        else:
            print(f"{row['label']} | {row['layer']} | failed | {row.get('error', 'unknown error')}")
    records.sort(key=lambda record: record[0]["latency_us"])
    for row, node in records:
        parent = row.get("parent") or "root"
        print(f"{row['label']} | {row['layer']} | {row['latency_us']:.3f} us | parent {parent}")
        for path in sorted((node / "artifacts").glob("*.sass")):
            print(f"  {path.stem}: {path.with_suffix('.ptx').relative_to(root)} -> {path.relative_to(root)}")
        if row["layer"] == "cuda":
            for path in sorted((node / "source").glob("*.cu")):
                print(f"  CUDA source: {path.relative_to(root)}")
    if any(row["layer"] != "tilelang" for row, _ in records):
        print("Source lookup: python scripts/source_map.py LABEL --sass-offset OFFSET "
              "(or --cuda-line LINE / --ptx-line LINE)")
    if not records:
        return
    selected = records[:1]
    if records[0][0]["layer"] == "tilelang":
        machine = next((record for record in records if record[0]["layer"] != "tilelang"), None)
        if machine is not None:
            selected.append(machine)
    settings = json.loads((root / "settings.json").read_text())
    for row, node in selected:
        try:
            profile(root, row, node, settings)
        except (RuntimeError, OSError) as error:
            print(f"CUDA profiling unavailable for {row['label']}: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
