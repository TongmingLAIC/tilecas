"""Report the retained candidates and profile the best measured launch graph."""

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
        with torch.inference_mode():
            inputs = reference.get_inputs()
            for _ in range(5):
                candidate(*inputs)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
                for _ in range(5):
                    candidate(*inputs)
                torch.cuda.synchronize()
        durations = defaultdict(float)
        for event in trace.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                durations[event.name] += event.time_range.elapsed_us() / 5
        if not durations:
            print("The CUDA profiler reported no device activities.")
            return
        print(f"CUDA activities for {row['label']} (mean microseconds per candidate call)")
        for name, elapsed in sorted(durations.items(), key=lambda item: -item[1]):
            print(f"  {elapsed:.3f} us | {name}")
        for index, spec in enumerate(program["kernels"] if program else []):
            callsite = spec.get("tilelang_callsite")
            origin = f" | TileLang {callsite['file']}:{callsite['line']}" if callsite else ""
            print(f"  kernel_{index}: {spec['entry']} | grid {spec['grid']} | block {spec['block']}"
                  f" | {spec['shared_bytes']} dynamic shared bytes{origin}")
    finally:
        if runtime is not None:
            runtime.close()


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
    if records:
        try:
            profile(root, *records[0], json.loads((root / "settings.json").read_text()))
        except (RuntimeError, OSError) as error:
            print(f"CUDA profiling unavailable: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
