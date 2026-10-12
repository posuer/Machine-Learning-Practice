"""离线验证训练函数的 loss/梯度契约；不模拟真实 VLM 的能力。"""

import copy
from types import SimpleNamespace

import pytest
import torch

from reference import train_step


class FixedPositionModel(torch.nn.Module):
    """可训练的逐位置分数表，方便独立推导期望 loss 和梯度。"""

    def __init__(self):
        super().__init__()
        self.scores = torch.nn.Parameter(
            torch.arange(30, dtype=torch.float32).reshape(1, 5, 6) / 10
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        assert pixel_values.device == input_ids.device == attention_mask.device == self.scores.device
        return SimpleNamespace(logits=self.scores.expand(input_ids.shape[0], -1, -1))


def make_batch():
    # token 含义：0=BOS，1=prompt，2/3=answer/EOS，4=PAD。
    return {
        "pixel_values": torch.zeros(1, 3, 2, 2),
        "input_ids": torch.tensor([[0, 1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        "labels": torch.tensor([[-100, -100, 2, 3, -100]]),
    }


def test_loss_gradients_update_and_batch_preservation():
    model = FixedPositionModel()
    model.eval()
    batch = make_batch()
    originals = {key: value.clone() for key, value in batch.items()}
    before = model.scores.detach().clone()
    # 根据题意独立选出两个预测：位置 1 -> token 2，位置 2 -> token 3。
    rows = before[0, [1, 2]]
    expected_loss = (rows.logsumexp(-1) - rows[torch.arange(2), torch.tensor([2, 3])]).mean()
    expected_grad = torch.zeros_like(before)
    for position, target in [(1, 2), (2, 3)]:
        expected_grad[0, position] = before[0, position].softmax(-1) / 2
        expected_grad[0, position, target] -= 0.5
    # 注入旧梯度，验证函数确实清理它们。
    model.scores.grad = torch.full_like(model.scores, 100)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    actual = train_step(model, batch, optimizer)
    assert isinstance(actual, float)
    assert actual == pytest.approx(expected_loss.item())
    assert model.training
    torch.testing.assert_close(model.scores.grad, expected_grad)
    torch.testing.assert_close(model.scores, before - 0.1 * expected_grad)
    for key in batch:
        torch.testing.assert_close(batch[key], originals[key])


def test_repeated_calls_do_not_accumulate_previous_gradients():
    model = FixedPositionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = make_batch()
    train_step(model, batch, optimizer)
    fresh = copy.deepcopy(model)
    fresh_optimizer = torch.optim.SGD(fresh.parameters(), lr=0.1)
    fresh_optimizer.zero_grad(set_to_none=True)
    train_step(model, batch, optimizer)
    train_step(fresh, batch, fresh_optimizer)
    torch.testing.assert_close(model.scores, fresh.scores)


def test_single_batch_loss_decreases():
    model = FixedPositionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    losses = [train_step(model, make_batch(), optimizer) for _ in range(40)]
    assert losses[-1] < losses[0] * 0.1


def test_empty_supervision_rejected_without_update():
    model = FixedPositionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = make_batch()
    batch["labels"].fill_(-100)
    before = model.scores.detach().clone()
    with pytest.raises(ValueError, match="no supervised"):
        train_step(model, batch, optimizer)
    torch.testing.assert_close(model.scores, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_cpu_batch_with_cuda_model():
    model = FixedPositionModel().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assert torch.isfinite(torch.tensor(train_step(model, make_batch(), optimizer)))
