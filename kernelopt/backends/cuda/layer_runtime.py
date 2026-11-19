"""Replay a static TileLang launch graph through the CUDA driver."""

import ctypes as C
import json
import os
from pathlib import Path
import shlex
import subprocess

import torch

from sass_edit import patch


SCALARS = {
    "float32": C.c_float, "float64": C.c_double,
    "int8": C.c_int8, "uint8": C.c_uint8, "bool": C.c_uint8,
    "int16": C.c_int16, "uint16": C.c_uint16,
    "int32": C.c_int32, "uint32": C.c_uint32,
    "int64": C.c_int64, "uint64": C.c_uint64,
}


def tool(name):
    root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    return str(Path(root) / "bin" / name) if root else name


def compile_ptx(source, output, spec, arch):
    from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH

    flags = ["--use_fast_math"] if spec["fast_math"] else []
    for option in spec["compile_flags"]:
        flags.extend(shlex.split(option))
    subprocess.run([
        tool("nvcc"), "-std=c++17", f"-arch={arch}", "--ptx", "-lineinfo", "-w",
        "-I", CUTLASS_INCLUDE_DIR, "-I", TILELANG_TEMPLATE_PATH,
        *flags, str(source), "-o", str(output),
    ], check=True)


def assemble(ptx, output, arch):
    subprocess.run([tool("ptxas"), f"-arch={arch}", str(ptx), "-o", str(output)], check=True)


def disassemble(cubin):
    return subprocess.check_output([tool("nvdisasm"), "-c", "-hex", str(cubin)], text=True)


def build(root, layer, artifacts):
    program = json.loads((root / "program.json").read_text())
    arch = program["arch"]
    artifacts.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(program["kernels"]):
        stem = f"kernel_{index}"
        ptx = artifacts / f"{stem}.ptx"
        cubin = artifacts / f"{stem}.cubin"
        if layer == "cuda":
            compile_ptx(root / f"{stem}.cu", ptx, spec, arch)
        else:
            ptx.write_bytes((root / f"{stem}.ptx").read_bytes())
        if layer == "sass":
            parent = root / f"{stem}.cubin"
            edits = json.loads((root / f"{stem}.sass.json").read_text())
            patch(parent, cubin, spec["entry"], arch, edits, disassemble)
        else:
            assemble(ptx, cubin, arch)
        (artifacts / f"{stem}.sass").write_text(disassemble(cubin))
    return program


class Driver:
    def __init__(self):
        self.library = C.CDLL("libcuda.so.1")
        self.error_name = self.library.cuGetErrorName
        self.error_name.argtypes = (C.c_int, C.POINTER(C.c_char_p))
        self.error_name.restype = C.c_int
        ptr = C.c_void_p
        uint = C.c_uint
        self.module_load = self.bind("cuModuleLoad", C.POINTER(ptr), C.c_char_p)
        self.module_unload = self.bind("cuModuleUnload", ptr)
        self.get_function = self.bind("cuModuleGetFunction", C.POINTER(ptr), ptr, C.c_char_p)
        self.attribute = self.bind("cuFuncSetAttribute", ptr, C.c_int, C.c_int)
        self.launch = self.bind("cuLaunchKernel", ptr, *([uint] * 7), ptr, C.POINTER(ptr), ptr)

    def bind(self, name, *argtypes):
        function = getattr(self.library, name)
        function.argtypes = argtypes
        function.restype = C.c_int

        def checked(*args):
            status = function(*args)
            if status:
                error = C.c_char_p()
                self.error_name(status, C.byref(error))
                details = error.value.decode() if error.value else str(status)
                raise RuntimeError(f"{name} failed: {details}")

        return checked


class Runtime:
    def __init__(self, program, artifacts):
        # A real allocation ensures PyTorch has made the primary context current.
        torch.empty(1, device="cuda")
        self.driver = Driver()
        self.program = program
        self.modules = []
        self.functions = []
        try:
            for index, spec in enumerate(program["kernels"]):
                module = C.c_void_p()
                self.driver.module_load(C.byref(module), os.fsencode(artifacts / f"kernel_{index}.cubin"))
                self.modules.append(module)
                entry = C.c_void_p()
                self.driver.get_function(C.byref(entry), module, spec["entry"].encode())
                if spec["shared_bytes"]:
                    self.driver.attribute(entry, 8, spec["shared_bytes"])
                self.functions.append(entry)
        except Exception:
            self.close()
            raise

    def close(self):
        if self.modules:
            torch.cuda.synchronize()
        while self.modules:
            self.driver.module_unload(self.modules.pop())

    def __call__(self, *inputs):
        values = dict(zip(self.program["inputs"], inputs, strict=True))
        stream = torch.cuda.current_stream()
        for entry, spec in zip(self.functions, self.program["kernels"], strict=True):
            arguments = []
            for param in spec["params"]:
                name, dtype = param["value"], param["dtype"]
                if param["kind"] == "scalar":
                    arguments.append(SCALARS[dtype](values[name]))
                    continue
                if name not in values:
                    if name in self.program["inputs"]:
                        raise ValueError(f"Missing tensor input {name}")
                    values[name] = torch.empty(param["shape"], dtype=getattr(torch, dtype), device=stream.device)
                tensor = values[name]
                if (list(tensor.shape) != param["shape"] or tensor.dtype != getattr(torch, dtype)
                        or tensor.device != stream.device or not tensor.is_contiguous()):
                    raise ValueError(f"{name} does not match the static CUDA tensor ABI")
                arguments.append(C.c_void_p(tensor.data_ptr()))
            pointers = (C.c_void_p * len(arguments))(
                *(C.cast(C.byref(value), C.c_void_p) for value in arguments)
            )
            self.driver.launch(entry, *spec["grid"], *spec["block"], spec["shared_bytes"],
                               stream.cuda_stream, pointers, None)
        return unpack(self.program["output"], values)


def unpack(layout, values):
    if isinstance(layout, str):
        return values[layout]
    result = [unpack(item, values) for item in layout["items"]]
    return tuple(result) if layout["type"] == "tuple" else result
