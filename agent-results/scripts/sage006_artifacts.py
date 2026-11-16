#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute the SAGE-005 artifact statistics from the compiled files.

The original report quoted opcode and IR counts that were produced in a shell
session and never committed.  This script re-derives them from the artefacts
themselves, so every number in the report has a command behind it:

* ``--ir``        op counts per IR level, by *operation*, not by substring.
* ``--vendor``    symbol size, instruction count and opcode histogram for one
                  ``flash_attn_fwd_kernel`` instantiation.
* ``--sha256``    digests of the vendor library and the extracted sections.

It never loads a device; only the toolchain is used.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import subprocess
import sys

TOOLCHAIN = "/opt/tops/bin"
LIB = "/lib/libkernel_flash_attn_fwd.so"
# Mangled names of the four instantiations in the shipped library.
SYMBOLS = {
    "bf16_256": "_ZN21topsop_global_kernels21flash_attn_fwd_kernelIu6__bf16Li256EEEvPT_PfS2_S2_S2_S3_S2_Pi24VARLEN_ATTN_FWD_OP_PARAS",
    "bf16_128": "_ZN21topsop_global_kernels21flash_attn_fwd_kernelIu6__bf16Li128EEEvPT_PfS2_S2_S2_S3_S2_Pi24VARLEN_ATTN_FWD_OP_PARAS",
    "fp16_256": "_ZN21topsop_global_kernels21flash_attn_fwd_kernelIDhLi256EEEvPT_PfS2_S2_S2_S3_S2_Pi24VARLEN_ATTN_FWD_OP_PARAS",
    "fp16_128": "_ZN21topsop_global_kernels21flash_attn_fwd_kernelIDhLi128EEEvPT_PfS2_S2_S2_S3_S2_Pi24VARLEN_ATTN_FWD_OP_PARAS",
}
ADDR = re.compile(r"^\s*[0-9a-f]+:")
# llvm-objdump separates the encoded bytes from the mnemonic with a tab.  A
# regex that instead takes the first token after the bytes picks up operand
# hex ("44", "18") whenever the line has no tab, which is how the SAGE-005
# report arrived at "173 distinct opcodes".
INSN = re.compile(r"^\s*[0-9a-f]+:.*?\t\s*([a-zA-Z][a-zA-Z0-9_.]*)")


def run(argv, **kwargs):
    return subprocess.run(argv, capture_output=True, text=True, check=False, **kwargs)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ir_counts(path):
    """Count IR operations by quoted op name.

    Substring counting is what produced the SAGE-005 ``8 gcu.mfence`` claim:
    each fence line also contains the ``gcu.mfence_type`` attribute name, so a
    naive ``str.count`` doubles the value.  Here the op name is matched only in
    its quoted form, and ``gcu.mfence_type`` therefore cannot match.
    """
    text = open(path, encoding="utf-8", errors="replace").read()
    counts = collections.Counter()
    for name in re.findall(r'"([a-zA-Z_][a-zA-Z0-9_.]*)"\s*\(', text):
        counts[name] += 1
    interesting = {
        key: counts[key]
        for key in ("gcu.matmul", "gcu.mfence", "memref.alloc", "memref.dealloc",
                    "gcu.dma", "memref.copy", "memref_ext.broadcast_start")
    }
    async_calls = collections.Counter(re.findall(r'callee = @(tops_[a-zA-Z0-9_]*async[a-zA-Z0-9_]*)', text))
    # The same names also occur inside attribute values (``triton_gcu.matmul``
    # borders, ``gcu.mfence_type``), so a plain ``str.count`` reports a larger
    # number.  Both are recorded to keep the historical figures traceable.
    substrings = {key: text.count(key) for key in ("gcu.matmul", "gcu.mfence")}
    pipeline = async_pipeline(text)
    return {
        "bytes": len(text.encode("utf-8")),
        "ops": interesting,
        "distinct_ops": len(counts),
        "async_helper_calls": dict(async_calls),
        "async_helper_total": sum(async_calls.values()),
        "substring_counts": substrings,
        "async_pipeline": pipeline,
    }


def async_pipeline(text):
    """Measure how far each async copy gets before its wait.

    ``tops_memcpy_async`` is present in the LLIR, so "no async copy" would be
    wrong.  What matters is whether the copy is ever left in flight: for every
    async call site, count the LLVM operations issued between it and the next
    ``tops_wait_dte``.  Zero means the pipeline is opened and closed in the same
    basic block, i.e. the transfer is synchronous in effect.
    """
    lines = text.splitlines()
    sites = []
    for index, line in enumerate(lines):
        if "callee = @tops_wait_dte" in line:
            continue
        match = re.search(r'callee = @(tops_[a-zA-Z0-9_]*async[a-zA-Z0-9_]*)', line)
        if not match:
            continue
        ops_until_wait = None
        for probe in range(index + 1, min(index + 400, len(lines))):
            if "callee = @tops_wait_dte" in lines[probe]:
                ops_until_wait = sum(
                    1 for inner in lines[index + 1:probe] if '"llvm.' in inner or '"gcu.' in inner)
                break
        sites.append(dict(callee=match.group(1), line=index + 1, ops_before_wait=ops_until_wait))
    waits = sum(1 for line in lines if "callee = @tops_wait_dte" in line)
    return {
        "call_sites": sites,
        "wait_calls": waits,
        "max_ops_before_wait": max((s["ops_before_wait"] or 0) for s in sites) if sites else None,
    }


def vendor_stats(workdir):
    os.makedirs(workdir, exist_ok=True)
    fatbin = os.path.join(workdir, "fa.fatbin")
    kernel = os.path.join(workdir, "fa.kernel")
    for out, section, extra in ((fatbin, ".tops_fatbin", ["-O", "binary"]), (kernel, None, None)):
        if os.path.exists(out):
            continue
        if section:
            proc = run([f"{TOOLCHAIN}/llvm-objcopy", LIB, out, "--only-section=" + section, *extra])
        else:
            proc = run([f"{TOOLCHAIN}/clang-offload-bundler", "--unbundle", "--type=o",
                        "--targets=tops-dtu-enflame-tops--gcu300",
                        "--input", fatbin, "--output", out])
        if proc.returncode != 0:
            raise SystemExit(f"extraction failed for {out}: {proc.stderr[:400]}")
    disasm = os.path.join(workdir, "fa.s")
    if not os.path.exists(disasm):
        with open(disasm, "w", encoding="utf-8") as handle:
            proc = subprocess.run([f"{TOOLCHAIN}/llvm-objdump", "-dz", "--triple=gcu300", kernel],
                                  stdout=handle, stderr=subprocess.PIPE, text=True, check=False)
        if proc.returncode != 0:
            raise SystemExit(f"objdump failed: {proc.stderr[:400]}")

    # The TOPs binutils bundle ships no llvm-nm; the host nm reads the ELF fine.
    nm = run(["nm", "-S", "--defined-only", kernel])
    sizes = {}
    for line in nm.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[3] in SYMBOLS.values():
            sizes[parts[3]] = int(parts[1], 16)

    lines = open(disasm, encoding="utf-8", errors="replace").read().splitlines()
    header = re.compile(r"^[0-9a-f]+ <(.+)>:$")
    out = []
    for label, symbol in SYMBOLS.items():
        start = end = None
        for index, line in enumerate(lines):
            match = header.match(line.strip())
            if not match:
                continue
            if match.group(1) == symbol:
                start = index
            elif start is not None:
                end = index
                break
        if start is None:
            continue
        segment = lines[start:end]
        histogram = collections.Counter()
        addr_lines = 0
        for line in segment:
            if not ADDR.match(line):
                continue
            addr_lines += 1
            match = INSN.match(line)
            if match:
                histogram[match.group(1)] += 1
        decoded = sum(histogram.values())
        nops = sum(value for key, value in histogram.items() if ".nop" in key)
        out.append(dict(
            kernel=label, symbol=symbol, symbol_bytes=sizes.get(symbol),
            text_bytes=len(segment), instruction_slots=addr_lines, decoded=decoded,
            undecoded=addr_lines - decoded, distinct_opcodes=len(histogram),
            nop_instructions=nops, opcodes=dict(histogram.most_common()),
        ))
    return out, dict(fatbin=sha256_file(fatbin), kernel=sha256_file(kernel),
                     disasm=sha256_file(disasm), library=sha256_file(LIB))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", help="comma-separated IR files to count")
    parser.add_argument("--vendor", action="store_true")
    parser.add_argument("--sha256", action="store_true")
    parser.add_argument("--workdir", default="/tmp/sage005_vendor")
    parser.add_argument("--jsonl")
    args = parser.parse_args()
    emit = None
    handle = None
    if args.jsonl:
        handle = open(args.jsonl, "w", encoding="utf-8")

    def write(record):
        if handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        print(json.dumps({k: v for k, v in record.items() if k != "opcodes"}, sort_keys=True))

    if args.ir:
        for path in [p for p in args.ir.split(",") if p]:
            write(dict(probe="ir", file=os.path.basename(path), **ir_counts(path)))
    if args.vendor:
        stats, digests = vendor_stats(args.workdir)
        for record in stats:
            write(dict(probe="vendor_kernel", **record))
        write(dict(probe="vendor_digest", **digests))
    if args.sha256:
        for path in (LIB,):
            write(dict(probe="sha256", file=path, sha256=sha256_file(path), bytes=os.path.getsize(path)))
    if handle:
        handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
