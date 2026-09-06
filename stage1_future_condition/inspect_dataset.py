"""Cheap dataset/protobuf preflight that does not instantiate visual models."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from .dataset import discover_chunks, source_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--video-name", default="256x256.mp4")
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    records = discover_chunks(args.root, args.video_name)
    split_chunks = Counter(source_split(x.source_id, args.seed) for x in records)
    split_sources = Counter()
    for source in {x.source_id for x in records}:
        split_sources[source_split(source, args.seed)] += 1
    print(json.dumps({
        "root": args.root,
        "chunks": len(records),
        "sources": len({x.source_id for x in records}),
        "chunks_by_split": split_chunks,
        "sources_by_split": split_sources,
    }, indent=2))


if __name__ == "__main__":
    main()
