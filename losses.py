import torch
from torch import nn
from torch.nn import functional as F


def dice_loss(prediction, target, dims):
    intersection = (prediction * target).sum(dim=dims)
    total = prediction.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - (2.0 * intersection + 1e-6) / (total + 1e-6)


def boundary_loss(probabilities, target):
    waveform = target[:, 1:].bool()
    onsets = waveform & ~F.pad(waveform[..., :-1], (1, 0))
    offsets = waveform & ~F.pad(waveform[..., 1:], (0, 1))
    events = torch.cat((onsets, offsets), dim=1).to(probabilities)

    difference = probabilities[:, 1:].diff(dim=-1)
    rising = F.pad(difference, (1, 0))
    falling = F.pad(-difference, (0, 1))
    strengths = (torch.cat((rising, falling), dim=1).relu() / 0.1).clamp(max=1.0)

    distance = torch.arange(-4, 5, device=events.device, dtype=events.dtype).abs()
    kernel = (1.0 - distance / 5).view(1, 1, -1).expand(6, 1, -1)
    boundary_targets = F.conv1d(events, kernel, padding=4, groups=6).clamp(max=1.0)
    weights = strengths.new_tensor([2, 1, 1, 2, 1, 1])
    return (dice_loss(strengths, boundary_targets, (0, 2)) * weights).sum() / weights.sum()


def cycle_loss(p_probability, labels):
    positions = torch.arange(labels.shape[-1], device=labels.device)
    qrs = labels == 3
    previous_qrs_end = torch.where(qrs, positions + 1, 0).cummax(dim=-1).values
    previous_qrs_end = F.pad(previous_qrs_end[:, :-1], (1, 0))
    onsets = qrs & ~F.pad(qrs[:, :-1], (1, 0))
    starts = torch.maximum(previous_qrs_end, positions - 64)
    valid = onsets & (positions >= 32) & (positions - starts >= 24)

    rows, anchors = valid.nonzero(as_tuple=True)
    left = starts[rows, anchors]
    offsets = torch.arange(-64, 0, device=labels.device)
    samples = anchors[:, None] + offsets
    inside = samples >= left[:, None]
    samples = samples.clamp_min(0)
    response = p_probability[rows[:, None], samples]
    scaled = (response / 0.25).masked_fill(~inside, -torch.inf)
    pooled = 0.25 * (scaled.logsumexp(dim=-1) - (anchors - left).to(response).log())
    pooled = pooled.clamp(0.0, 1.0)

    present = ((labels[rows[:, None], samples] == 1) & inside).any(dim=-1)
    squared_error = (pooled - present.to(pooled)).square()
    present_count = present.sum()
    absent_count = (~present).sum()
    present_mean = (squared_error * present).sum() / present_count.clamp_min(1)
    absent_mean = (squared_error * ~present).sum() / absent_count.clamp_min(1)
    weight = (present_count > 0).to(pooled) + 2 * (absent_count > 0).to(pooled)
    return (present_mean + 2 * absent_mean) / weight.clamp_min(1)


class CWLLoss(nn.Module):
    def forward(self, logits, labels):
        probabilities = logits.float().softmax(dim=1)
        target = F.one_hot(labels, num_classes=4).movedim(-1, 1).to(probabilities)
        region = dice_loss(probabilities, target, (0, 2)).mean()
        boundary = boundary_loss(probabilities, target)
        cycle = cycle_loss(probabilities[:, 1], labels)
        return region + 0.01 * boundary + 0.05 * cycle
