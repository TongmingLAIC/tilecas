"""Lower a validated candidate into editable device source, PTX or SASS edits."""

import argparse
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import torch

from layer_runtime import build, SCALARS
from module_loader import load
from sass_edit import instructions


RUN = Path(__file__).resolve().parents[1]
LAYERS = ("tilelang", "cuda", "ptx", "sass")


def make_model(source, reference):
    module = load(source, "tilecas_candidate")
    model = module.ModelNew(*reference.get_init_inputs())
    if hasattr(model, "to"):
        model = model.to("cuda")
    if hasattr(model, "eval"):
        model = model.eval()
    return model


def export_tilelang(source, reference, destination):
    import tilelang
    from tilelang.jit.kernel import JITKernel

    # Cache-restored adapters can omit the lowered device IR and its ABI.
    tilelang.disable_cache()
    torch.manual_seed(42)
    inputs = reference.get_inputs()
    names = {id(value): f"input_{index}" for index, value in enumerate(inputs)}
    if len(names) != len(inputs):
        raise ValueError("TileLang lowering requires uniquely identifiable top-level arguments")
    program = {"inputs": [names[id(value)] for value in inputs], "kernels": []}
    keepalive = list(inputs)
    original = JITKernel.__call__

    def capture(kernel, *args, **kwargs):
        if kwargs:
            raise ValueError("CUDA lowering requires positional kernel arguments")
        output = original(kernel, *args)
        results = [] if output is None else list(output) if isinstance(output, (tuple, list)) else [output]
        for value in results:
            names.setdefault(id(value), f"value_{len(names)}")
        keepalive.extend(results)
        source_text = kernel.get_kernel_source()
        spec = kernel_spec(kernel, args, results, names)
        for frame in inspect.stack(context=0):
            if Path(frame.filename).is_relative_to(source):
                spec["tilelang_callsite"] = {
                    "file": str(Path(frame.filename).relative_to(source)), "line": frame.lineno,
                }
                break
        index = len(program["kernels"])
        destination.joinpath(f"kernel_{index}.cu").write_text(source_text)
        arch = str(kernel.target.attrs["arch"])
        if program.setdefault("arch", arch) != arch:
            raise ValueError("All kernels must target the same GPU architecture")
        program["kernels"].append(spec)
        return output

    JITKernel.__call__ = capture
    try:
        with torch.inference_mode():
            output = make_model(source / "ModelNew.py", reference)(*inputs)
            torch.cuda.synchronize()
    finally:
        JITKernel.__call__ = original

    def pack(value):
        if isinstance(value, (list, tuple)):
            return {"type": type(value).__name__, "items": [pack(item) for item in value]}
        if id(value) not in names:
            raise ValueError("Output does not come from a captured kernel")
        return names[id(value)]

    if not program["kernels"]:
        raise ValueError("No TileLang launches were captured")
    program["output"] = pack(output)
    destination.joinpath("program.json").write_text(json.dumps(program, indent=2) + "\n")


def kernel_spec(kernel, args, results, names):
    adapter = kernel.adapter
    if adapter.device_mod is None or len(adapter.device_mod.functions) != 1:
        raise ValueError("CUDA lowering requires one device entry per TileLang JIT call")
    _, device = next(iter(adapter.device_mod.functions.items()))
    attrs = device.attrs
    unsupported = {"cluster_dims", "use_cooperative_groups", "has_cuda_pdl_sync"}
    if unsupported.intersection(str(key) for key in attrs.keys()):
        raise ValueError("CUDA lowering requires a regular, noncooperative launch")

    prim = adapter.prim_func
    returned = dict(zip(adapter.result_idx, results, strict=True))
    incoming = iter(args)
    params = {}
    for index, param in enumerate(prim.params):
        value = returned[index] if index in returned else next(incoming)
        if id(value) not in names:
            raise ValueError("Kernel argument does not come from a reference input or an earlier TileLang call")
        spec = {"value": names[id(value)]}
        if param in prim.buffer_map:
            buffer = prim.buffer_map[param]
            if not isinstance(value, torch.Tensor) or not value.is_cuda or not value.is_contiguous():
                raise ValueError("CUDA lowering requires contiguous GPU tensors")
            if [int(extent) for extent in buffer.shape] != list(value.shape):
                raise ValueError("CUDA lowering requires static tensor shapes")
            spec.update(kind="tensor", shape=list(value.shape), dtype=str(value.dtype).removeprefix("torch."))
            params[buffer.data.name] = spec
        else:
            dtype = str(param.dtype)
            if dtype not in SCALARS or not isinstance(value, (int, float, bool)):
                raise ValueError(f"Unsupported scalar ABI: {dtype}")
            spec.update(kind="scalar", dtype=dtype)
            params[param.name] = spec

    grid, block = [1, 1, 1], [1, 1, 1]
    for tag, extent in attrs.get("thread_extent", {}).items():
        tag = str(tag)
        if tag.startswith("blockIdx."):
            grid["xyz".index(tag[-1])] = int(extent)
        if tag.startswith("threadIdx."):
            block["xyz".index(tag[-1])] = int(extent)
    return {
        "entry": str(attrs["global_symbol"]), "grid": grid, "block": block,
        "shared_bytes": int(attrs.get("dyn_shared_memory_buf", 0)),
        "params": [params[param.name] for param in device.params],
        "fast_math": bool((kernel.pass_configs or {}).get("tl.enable_fast_math")),
        "compile_flags": list(kernel.compile_flags or []),
    }


def lower(parent, destination, target=None):
    result = json.loads(parent.joinpath("result.json").read_text())
    if not result["correct"]:
        raise ValueError("Select a measured, correct parent")
    layer = result["layer"]
    start = LAYERS.index(layer)
    if start == len(LAYERS) - 1:
        raise ValueError("SASS is the lowest editable layer")
    finish = LAYERS.index(target) if target else start + 1
    if finish <= start:
        raise ValueError("Select a lower layer or branch from a retained higher-layer candidate")
    allowed = json.loads((RUN / "settings.json").read_text())["layers"]
    if LAYERS[finish] not in allowed:
        raise ValueError(f"{LAYERS[finish]} is disabled in this run")

    current = parent / "source"
    artifacts = parent / "artifacts"
    for index in range(start, finish):
        outgoing = destination if index + 1 == finish else destination.parent / LAYERS[index + 1]
        outgoing.mkdir(exist_ok=True)
        if LAYERS[index] == "tilelang":
            reference = load(RUN / "reference.py", "tilecas_reference")
            export_tilelang(current, reference, outgoing)
        else:
            shutil.copyfile(current / "program.json", outgoing / "program.json")
            if LAYERS[index] == "cuda":
                archived = outgoing / "cuda-source"
                archived.mkdir()
                for path in current.glob("kernel_*.cu"):
                    shutil.copyfile(path, archived / path.name)
            elif (current / "cuda-source").is_dir():
                shutil.copytree(current / "cuda-source", outgoing / "cuda-source")
            for path in artifacts.glob("*.ptx"):
                shutil.copyfile(path, outgoing / path.name)
                if LAYERS[index] == "ptx":
                    for extension in (".cubin", ".sass"):
                        sibling = path.with_suffix(extension)
                        shutil.copyfile(sibling, outgoing / sibling.name)
                    outgoing.joinpath(path.stem + ".sass.json").write_text("[]\n")
        current = outgoing
        if index + 1 < finish:
            artifacts = destination.parent / (LAYERS[index + 1] + "-build")
            build(current, LAYERS[index + 1], artifacts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", help="A successful trajectory label")
    parser.add_argument("candidate", help="Name of the new editable candidate")
    parser.add_argument("--to", choices=LAYERS[1:])
    args = parser.parse_args()
    if not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) for name in (args.parent, args.candidate)):
        parser.error("Use letters, digits, underscores and hyphens in candidate and trajectory names")
    settings = json.loads((RUN / "settings.json").read_text())
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(settings["gpu"])
    root = RUN / "candidates"
    destination = root / args.candidate
    if destination.exists():
        parser.error(f"Candidate already exists: {destination}")
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        payload = Path(tmp) / "source"
        payload.mkdir()
        lower(RUN / "trajectory" / args.parent, payload, args.to)
        payload.joinpath("parent.txt").write_text(args.parent + "\n")
        payload.rename(destination)
    print(destination)


if __name__ == "__main__":
    main()
