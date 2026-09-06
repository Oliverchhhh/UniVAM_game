"""Load generated protobufs without importing the heavyweight elefant.data package."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_video_annotation_pb2():
    proto_dir = Path(__file__).resolve().parents[1] / "elefant" / "data" / "proto"
    # Importing elefant.data normally imports the Rust extension and the full
    # training stack. Stage I only needs these two generated protobuf modules.
    if "elefant" not in sys.modules:
        elefant = types.ModuleType("elefant")
        elefant.__path__ = [str(proto_dir.parents[1])]
        sys.modules["elefant"] = elefant
    if "elefant.data" not in sys.modules:
        data = types.ModuleType("elefant.data")
        data.__path__ = [str(proto_dir.parent)]
        sys.modules["elefant.data"] = data
    if "elefant.data.proto" not in sys.modules:
        package = types.ModuleType("elefant.data.proto")
        package.__path__ = [str(proto_dir)]
        sys.modules["elefant.data.proto"] = package
    shared_name = "elefant.data.proto.shared_pb2"
    shared = sys.modules.get(shared_name) or _load_module(shared_name, proto_dir / "shared_pb2.py")
    sys.modules["elefant.data.proto"].shared_pb2 = shared
    video_name = "elefant.data.proto.video_annotation_pb2"
    video = sys.modules.get(video_name) or _load_module(video_name, proto_dir / "video_annotation_pb2.py")
    sys.modules["elefant.data.proto"].video_annotation_pb2 = video
    return video


video_annotation_pb2 = _load_video_annotation_pb2()
