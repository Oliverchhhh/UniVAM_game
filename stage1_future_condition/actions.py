"""Cuphead protobuf gamepad actions -> NitroGen's continuous 25-D layout."""

from __future__ import annotations

import torch

ACTION_HORIZON = 18
ACTION_DIM = 25

# This order is the released NitroGen tokenizer/checkpoint order.  The final
# four dimensions are [left_x, left_y, right_x, right_y], normalized to [0, 1].
BUTTON_NAMES = (
    "BACK",
    "DPAD_DOWN",
    "DPAD_LEFT",
    "DPAD_RIGHT",
    "DPAD_UP",
    "EAST",
    "GUIDE",
    "LEFT_SHOULDER",
    "LEFT_THUMB",
    "LEFT_TRIGGER",
    "NORTH",
    "RIGHT_BOTTOM",
    "RIGHT_LEFT",
    "RIGHT_RIGHT",
    "RIGHT_SHOULDER",
    "RIGHT_THUMB",
    "RIGHT_TRIGGER",
    "RIGHT_UP",
    "SOUTH",
    "START",
    "WEST",
)

_PROTO_BUTTON = {
    "BACK": "select",
    "DPAD_DOWN": "dpad_down",
    "DPAD_LEFT": "dpad_left",
    "DPAD_RIGHT": "dpad_right",
    "DPAD_UP": "dpad_up",
    "EAST": "east",
    "LEFT_SHOULDER": "left_bumper",
    "NORTH": "north",
    "RIGHT_SHOULDER": "right_bumper",
    "SOUTH": "south",
    "START": "start",
    "WEST": "west",
}

# GUIDE, the four RIGHT_* directional aliases, and digital trigger buttons do
# not exist in the Open-P2P protobuf.  Masking them is crucial: treating them as
# negative examples would teach a false target distribution.
_SUPPORTED_BUTTONS = set(_PROTO_BUTTON) | {"LEFT_THUMB", "RIGHT_THUMB"}
SUPPORTED_ACTION_MASK = torch.tensor(
    [name in _SUPPORTED_BUTTONS for name in BUTTON_NAMES] + [True] * 4,
    dtype=torch.bool,
)


def _axis_to_unit(value: float) -> float:
    """Clip a joystick axis to [-1, 1] and normalize it to [0, 1]."""
    return (max(-1.0, min(1.0, float(value))) + 1.0) * 0.5


def gamepad_to_nitrogen(gamepad) -> torch.Tensor:
    """Convert a GamePadAction protobuf to one float32 vector of shape [25]."""
    out = torch.zeros(ACTION_DIM, dtype=torch.float32)
    for idx, name in enumerate(BUTTON_NAMES):
        if name == "LEFT_THUMB":
            out[idx] = float(gamepad.left_stick.pressed)
        elif name == "RIGHT_THUMB":
            out[idx] = float(gamepad.right_stick.pressed)
        elif name in _PROTO_BUTTON:
            out[idx] = float(getattr(gamepad.buttons, _PROTO_BUTTON[name]))

    out[-4:] = torch.tensor(
        [
            _axis_to_unit(gamepad.left_stick.x),
            _axis_to_unit(gamepad.left_stick.y),
            _axis_to_unit(gamepad.right_stick.x),
            _axis_to_unit(gamepad.right_stick.y),
        ],
        dtype=torch.float32,
    )
    return out


def frame_action(frame_annotation) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefer a known system action, then user action; otherwise return masked no-op."""
    source = None
    if frame_annotation.system_action.is_known and frame_annotation.system_action.HasField("game_pad"):
        source = frame_annotation.system_action.game_pad
    elif frame_annotation.user_action.is_known and frame_annotation.user_action.HasField("game_pad"):
        source = frame_annotation.user_action.game_pad

    if source is None:
        return torch.zeros(ACTION_DIM, dtype=torch.float32), torch.zeros(ACTION_DIM, dtype=torch.bool)
    return gamepad_to_nitrogen(source), SUPPORTED_ACTION_MASK.clone()
