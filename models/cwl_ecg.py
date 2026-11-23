import torch
from torch import nn
from torch.nn import functional as F


def conv_block(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.GELU(),
    )


class MultiScaleLocal(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = 2 * channels
        self.expand = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False), nn.BatchNorm1d(hidden), nn.ReLU6()
        )
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2, groups=hidden, bias=False),
                nn.BatchNorm1d(hidden),
                nn.ReLU6(),
            )
            for kernel in (3, 5, 7)
        )
        self.shuffle = nn.ChannelShuffle(2)
        self.project = nn.Sequential(
            nn.Conv1d(hidden, channels, 1, bias=False), nn.BatchNorm1d(channels)
        )

    def forward(self, x):
        expanded = self.expand(x)
        mixed = torch.stack([branch(expanded) for branch in self.branches]).sum(dim=0)
        return x + self.project(self.shuffle(mixed))


class BidirectionalMamba2(nn.Module):
    def __init__(self, channels, headdim):
        super().__init__()
        from mamba_ssm import Mamba2

        self.norm = nn.LayerNorm(channels)
        self.forward_mamba = Mamba2(channels, d_state=32, d_conv=4, expand=1, headdim=headdim)
        self.reverse_mamba = Mamba2(channels, d_state=32, d_conv=4, expand=1, headdim=headdim)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        sequence = x.transpose(1, 2)
        normalized = self.norm(sequence)
        forward = self.forward_mamba(normalized)
        reverse = self.reverse_mamba(normalized.flip(dims=(1,))).flip(dims=(1,))
        return (sequence + self.dropout(0.5 * (forward + reverse))).transpose(1, 2)


class EncoderStage(nn.Module):
    def __init__(self, in_channels, channels, stride, headdim):
        super().__init__()
        self.entry = conv_block(in_channels, channels, stride=stride)
        self.local = MultiScaleLocal(channels)
        self.context = BidirectionalMamba2(channels, headdim)

    def forward(self, x):
        return self.context(self.local(self.entry(x)))


class CrossGatedSkip(nn.Module):
    def __init__(self, channels):
        super().__init__()

        def gate():
            return nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=1, groups=channels),
                nn.Conv1d(channels, channels, 1),
                nn.Sigmoid(),
            )

        self.encoder_gate = gate()
        self.decoder_gate = gate()
        self.mask = nn.Sequential(nn.Conv1d(channels, channels, 1), nn.Sigmoid())

    def forward(self, encoder, decoder):
        crossed = self.encoder_gate(encoder) * decoder + self.decoder_gate(decoder) * encoder
        return encoder * self.mask(crossed)


class DecoderStage(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.refine = nn.Sequential(conv_block(in_channels, channels), conv_block(channels, channels))

    def forward(self, x, length):
        x = F.interpolate(x, size=length, mode="linear", align_corners=False)
        return self.refine(x)


class CWLECG(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList((
            EncoderStage(1, 32, 1, 32),
            EncoderStage(32, 64, 2, 64),
            EncoderStage(64, 128, 2, 64),
            EncoderStage(128, 256, 2, 64),
        ))
        self.decoders = nn.ModuleList((
            DecoderStage(256, 128), DecoderStage(128, 64), DecoderStage(64, 32)
        ))
        self.skip_gates = nn.ModuleList((CrossGatedSkip(128), CrossGatedSkip(64)))
        self.head = nn.Conv1d(32, 4, 1)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            features = []
            for stage in self.encoder:
                x = stage(x)
                features.append(x)
            for index, (stage, skip) in enumerate(zip(self.decoders, reversed(features[:-1]))):
                guide = stage(x, skip.shape[-1])
                if index < len(self.skip_gates):
                    skip = self.skip_gates[index](skip, guide)
                x = guide + skip
            return self.head(x)
