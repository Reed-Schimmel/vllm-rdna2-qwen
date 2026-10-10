#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare a lossless INT4 flat table. This reads and writes the whole table."""

import argparse

from vllm.v1.ple_offload.int4_nvme import prepare_int4_nvme_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sidecar", required=True, help="Directory containing META.json"
    )
    parser.add_argument("--output", required=True, help="Dedicated real NVMe directory")
    parser.add_argument("--chunk-rows", type=int, default=65536)
    args = parser.parse_args()
    table = prepare_int4_nvme_table(args.sidecar, args.output, args.chunk_rows)
    print(f"Prepared INT4 NVMe table: {table}")


if __name__ == "__main__":
    main()
