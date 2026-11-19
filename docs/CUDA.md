# TileCas on CUDA

The CUDA launcher runs one optimizer conversation across TileLang, CUDA C++, PTX, and SASS. The optimizer selects the layer and can return to a retained branch. Each candidate is replayed against the same reference and ranked by median CUDA Event latency across three seeds.

## Launch

Use a Linux NVIDIA host with PyTorch, TileLang, the CUDA toolkit, binutils, bubblewrap, and an authenticated Codex CLI. Activate the GPU Python environment first. Run on an available GPU.

```sh
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"
python scripts/run_cuda_tilecas.py "$HOME/workspace/runs/cuda-example" \
  --gpu 0 --reference /path/to/reference.py --tilelang /path/to/ModelNew.py \
  --model gpt-6.2-sol --effort low
```

The reference supplies `Model`, `get_inputs()`, and `get_init_inputs()`. Generate random floating inputs with `torch.randn`. To include a small file that the reference loads next to itself, pass `--asset /path/to/file`. For a manual run, add `--stage-only` and follow `AGENTS.md` in the new workspace. For a three-layer comparison, add `--layers tilelang cuda ptx`. Use the same inputs, GPU and timing settings for each comparison.

## Edit and evaluate

| Layer | Editable files | Build and launch |
| --- | --- | --- |
| TileLang | `ModelNew.py` | TileLang JIT |
| CUDA C++ | `kernel_N.cu`, `program.json` | `nvcc` emits PTX, then `ptxas` builds a cubin |
| PTX | `kernel_N.ptx`, `program.json` | `ptxas` builds a cubin |
| SASS | `kernel_N.sass.json`, with the disassembly in `kernel_N.sass` | Apply instruction edits to the parent cubin and disassemble it again before launch |

```sh
bash scripts/bench.sh tilelang initial baseline
python scripts/lower_layers.py baseline native
bash scripts/bench.sh cuda native native-baseline
python scripts/lower_layers.py native-baseline assembly
bash scripts/bench.sh ptx assembly ptx-baseline
python scripts/lower_layers.py ptx-baseline machine
# Edit candidates/machine/kernel_0.sass.json to make a SASS candidate.
bash scripts/bench.sh sass machine sass-iter-1
python scripts/diagnose.py
```

Each label retains source, parent, correctness, latency, and available compiler artifacts in `trajectory/`. Use `--to` with the lowering command to skip layers. To revisit a node, copy its source to a new candidate directory and set `parent.txt` to the original label.

Automatic lowering covers static contiguous tensors, scalar parameters, and standard launches with one device entry per JIT call. Multiple sequential calls are captured. The native launch graph preserves argument order and uses the current PyTorch CUDA stream. The Hopper editor supports validated adjacent ALU swaps and increasing an ALU instruction's stall field. The disassembled instruction and cubin encoding are checked against each other before and after every patch.

## Source correspondence

Use the exact cubin from the measured node to trace CUDA and PTX lines to machine
instruction offsets, or look up where a machine instruction came from.

```sh
python scripts/source_map.py native-baseline --kernel 0 --cuda-line 32
python scripts/source_map.py ptx-baseline --kernel 0 --ptx-line 57
python scripts/source_map.py sass-iter-1 --kernel 0 --sass-offset 0x80
```

CUDA and PTX locations come from the compiler's line tables. Lowered nodes
retain a copy of their CUDA source, including when lowering skips a layer. SASS swaps carry
their original locations with the edited instructions. A TileLang location
identifies the Python call that launched the corresponding kernel. Inlining,
instruction fusion and scheduling can associate several machine instructions
with the same source line, and some instructions have no reported location.
