"""Apply verified, size-preserving edits to Hopper cubin instructions."""

import re
import struct
import subprocess


SECTION = re.compile(r"^\s*\.section\s+\.text\.([^,\s]+),")
INSTRUCTION = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+(.+?);\s*/\* (0x[0-9a-f]+) \*/")
HIGH_WORD = re.compile(r"\s*/\* (0x[0-9a-f]+) \*/\s*$")
LOCAL_LABEL = re.compile(r"^\.L[^:]*:")
REGISTER = r"R(?:[0-9]+|Z)"
ALU = re.compile(rf"^(FADD|FMUL|FFMA|MOV)\s+({REGISTER}),\s*(.*)$")
STALL = 0xf << 41
SCHEDULING = ((1 << 23) - 1) << 41


def instructions(text, entry):
    selected = False
    block = 0
    result = {}
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if match := SECTION.match(line):
            selected = match[1] == entry
        if not selected:
            continue
        if LOCAL_LABEL.match(line.strip()):
            block += 1
        if match := INSTRUCTION.match(line):
            high = HIGH_WORD.fullmatch(lines[index + 1])
            if high is None:
                raise ValueError("Expected Hopper 128-bit instruction encoding")
            result[int(match[1], 16)] = (match[2].strip(), int(match[3], 16), int(high[1], 16), block)
    if not result:
        raise ValueError(f"No disassembled instructions for {entry}")
    return result


def section_offset(cubin, entry):
    sections = subprocess.check_output(["readelf", "-SW", str(cubin)], text=True)
    row = re.search(rf"^\s*\[\s*\d+\]\s+\.text\.{re.escape(entry)}\s+PROGBITS"
                    r"\s+[0-9a-f]+\s+([0-9a-f]+)\s+([0-9a-f]+)\b", sections, re.M)
    if row is None:
        raise ValueError(f"Missing ELF section .text.{entry}")
    return int(row[1], 16), int(row[2], 16)


def alu_registers(text):
    match = ALU.fullmatch(text)
    if match is None or match[2] == "RZ":
        raise ValueError("This edit requires an unpredicated scalar MOV, FADD, FMUL or FFMA")
    reads = set()
    for operand in match[3].split(","):
        operand = operand.strip()
        if not re.fullmatch(rf"-?(?:{REGISTER}|0x[0-9a-f]+|[0-9]+(?:\.[0-9]*)?)", operand):
            raise ValueError(f"Unsupported operand in SASS edit: {operand}")
        if re.fullmatch(REGISTER, operand.lstrip("-")) and operand.lstrip("-") != "RZ":
            reads.add(operand.lstrip("-"))
    return match[1], match[2], reads


def swap_pair(first, second):
    a_op, a_write, a_read = alu_registers(first[0])
    b_op, b_write, b_read = alu_registers(second[0])
    if (a_op != b_op or first[3] != second[3]
            or a_write == b_write or a_write in b_read or b_write in a_read
            or (first[2] & SCHEDULING) != (second[2] & SCHEDULING)
            or (first[2] >> 58) != 0 or (first[2] & STALL) >> 41 < 8):
        raise ValueError("The swap requires independent same-opcode instructions in one basic block, "
                         "with identical scheduling fields, no operand reuse and at least eight stall cycles")
    return second[1:3], first[1:3]


def patch(parent, output, entry, arch, edits, disassemble):
    if arch not in {"sm_90", "sm_90a"}:
        raise ValueError("SASS edits require a Hopper target")
    original = instructions(disassemble(parent), entry)
    offset, size = section_offset(parent, entry)
    data = bytearray(parent.read_bytes())
    expected = {}
    for edit in edits:
        address = int(edit["offset"], 0)
        if edit["op"] == "swap":
            peer = address + 16
            first, second = original[address], original[peer]
            a, b = swap_pair(first, second)
            changes = {address: (a, second[0]), peer: (b, first[0])}
        elif edit["op"] == "stall":
            text, low, high, _ = original[address]
            alu_registers(text)
            stall = (high & STALL) >> 41
            cycles = edit["cycles"]
            if type(cycles) is not int or not stall < cycles <= 15:
                raise ValueError("Stall edits may increase the encoded delay up to 15 cycles")
            changes = {address: ((low, (high & ~STALL) | (cycles << 41)), text)}
        else:
            raise ValueError(f"Unsupported SASS edit: {edit['op']}")
        for position, (words, assembly) in changes.items():
            begin = offset + position
            if (position in expected or position % 16 or not 0 <= position <= size - 16
                    or tuple(struct.unpack_from("<QQ", data, begin)) != original[position][1:3]):
                raise ValueError(f"Overlapping or invalid cubin instruction at {position:#x}")
            struct.pack_into("<QQ", data, begin, *words)
            expected[position] = (assembly, *words)
    output.write_bytes(data)
    observed = instructions(disassemble(output), entry)
    for position, instruction in expected.items():
        if observed[position][:3] != instruction:
            raise ValueError(f"Patched instruction did not round-trip at {position:#x}")
