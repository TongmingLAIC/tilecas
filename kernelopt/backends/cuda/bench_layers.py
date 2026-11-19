"""Measure a four-layer CUDA candidate against the run's reference."""

import argparse
import json
import re
import shutil
import statistics
import time

import torch

from layer_runtime import build, Runtime
from lower_layers import RUN, LAYERS, make_model
from module_loader import load
from validate_candidate import validate


SEEDS = (42, 43, 50000)


def latency(model, inputs):
    for _ in range(10):
        model(*inputs)
    torch.cuda.synchronize()
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(50)]
    for start, end in pairs:
        start.record()
        model(*inputs)
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in pairs) * 1000


def correctness(output, expected, atol, rtol):
    if isinstance(expected, (tuple, list)):
        if not isinstance(output, (tuple, list)) or len(output) != len(expected):
            raise ValueError("Unexpected output structure")
        for actual, reference in zip(output, expected):
            correctness(actual, reference, atol, rtol)
    else:
        torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)


@torch.inference_mode()
def evaluate(layer, root, artifacts, settings):
    torch.set_float32_matmul_precision("highest")
    reference = load(RUN / "reference.py", "tilecas_reference")
    baseline = reference.Model(*reference.get_init_inputs()).cuda().eval()
    runtime = None
    if layer == "tilelang":
        errors = validate(root / "ModelNew.py", "High", root)
        if errors:
            raise ValueError("\n".join(errors))
        candidate = make_model(root / "ModelNew.py", reference)
    else:
        if layer == "cuda":
            from validate_candidate import cuda_template_errors
            for path in root.glob("*.cu"):
                if errors := cuda_template_errors(path.read_text(), path, root):
                    raise ValueError("\n".join(errors))
        runtime = Runtime(build(root, layer, artifacts), artifacts)
        candidate = runtime

    samples = []
    try:
        for seed in SEEDS:
            torch.manual_seed(seed)
            inputs = reference.get_inputs()
            expected = baseline(*inputs)
            correctness(candidate(*inputs), expected, settings["atol"], settings["rtol"])
            measured = latency(candidate, inputs)
            ref_us = latency(baseline, inputs)
            samples.append({"seed": seed, "latency_us": measured, "reference_us": ref_us})
    finally:
        if runtime is not None:
            runtime.close()
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layer", choices=LAYERS)
    parser.add_argument("candidate")
    parser.add_argument("label")
    args = parser.parse_args()
    if not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) for name in (args.candidate, args.label)):
        parser.error("Use letters, digits, underscores and hyphens in candidate and trajectory names")
    settings = json.loads((RUN / "settings.json").read_text())
    if args.layer not in settings["layers"]:
        parser.error(f"{args.layer} is disabled in this run")
    start = time.monotonic()
    destination = RUN / "trajectory" / args.label
    destination.mkdir()
    snapshot = destination / "source"
    source = RUN / "candidates" / args.candidate
    result = {"label": args.label, "layer": args.layer, "candidate": args.candidate, "correct": False}
    try:
        shutil.copytree(source, snapshot, ignore=shutil.ignore_patterns("__pycache__", ".DS_Store"))
        parent = snapshot / "parent.txt"
        result["parent"] = parent.read_text().strip() if parent.exists() else None
        samples = evaluate(args.layer, snapshot, destination / "artifacts", settings)
        measured = statistics.median(row["latency_us"] for row in samples)
        ref_us = statistics.median(row["reference_us"] for row in samples)
        result.update(correct=True, latency_us=measured, reference_us=ref_us,
                      speedup=ref_us / measured, samples=samples, timing="cuda_events")
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        result["wall_seconds"] = round(time.monotonic() - start, 3)
        destination.joinpath("result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
