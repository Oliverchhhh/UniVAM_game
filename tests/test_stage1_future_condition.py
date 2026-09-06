import torch

from stage1_future_condition.actions import ACTION_DIM, BUTTON_NAMES, SUPPORTED_ACTION_MASK, frame_action, gamepad_to_nitrogen
from stage1_future_condition.dataset import source_split
from stage1_future_condition.model import ConditionCrossAttentionAdapter, FutureQFormer
from stage1_future_condition.proto import video_annotation_pb2


def test_action_layout_and_mask():
    gamepad = video_annotation_pb2.GamePadAction()
    gamepad.buttons.south = True
    gamepad.buttons.dpad_left = True
    gamepad.left_stick.x = -1.0
    gamepad.left_stick.y = 0.0
    gamepad.right_stick.x = 1.0
    gamepad.right_stick.y = 0.5
    action = gamepad_to_nitrogen(gamepad)
    assert action.shape == (ACTION_DIM,)
    assert action[BUTTON_NAMES.index("SOUTH")] == 1
    assert action[BUTTON_NAMES.index("DPAD_LEFT")] == 1
    assert torch.allclose(action[-4:], torch.tensor([0.0, 0.5, 1.0, 0.75]))
    assert not SUPPORTED_ACTION_MASK[BUTTON_NAMES.index("GUIDE")]
    assert not SUPPORTED_ACTION_MASK[BUTTON_NAMES.index("LEFT_TRIGGER")]
    assert SUPPORTED_ACTION_MASK[-4:].all()


def test_frame_action_prefers_system_and_masks_unknown():
    frame = video_annotation_pb2.FrameAnnotation()
    action, mask = frame_action(frame)
    assert not mask.any() and not action.any()
    frame.user_action.is_known = True
    frame.user_action.game_pad.buttons.south = True
    frame.system_action.is_known = True
    frame.system_action.game_pad.buttons.north = True
    action, mask = frame_action(frame)
    assert action[BUTTON_NAMES.index("NORTH")] == 1
    assert action[BUTTON_NAMES.index("SOUTH")] == 0
    assert mask.any()


def test_source_split_is_stable():
    assert source_split("v123", 43) == source_split("v123", 43)
    assert source_split("v123", 43) in {"train", "val", "test"}


def test_qformer_and_adapter_shapes_and_gradients():
    qformer = FutureQFormer(hidden_dim=64, num_heads=8, ff_dim=128, num_layers=2, output_dim=16, wan_dim=49, dino_dim=64)
    adapter = ConditionCrossAttentionAdapter(model_dim=64, condition_dim=16, inner_dim=32, heads=8)
    condition = qformer(torch.randn(2, 24, 64), torch.randn(2, 8, 49))
    hidden = adapter(torch.randn(2, 18, 64), condition)
    assert condition.shape == (2, 18, 16)
    assert hidden.shape == (2, 18, 64)
    hidden.square().mean().backward()
    assert qformer.query_tokens.grad is not None
    assert adapter.out_proj.weight.grad is not None
