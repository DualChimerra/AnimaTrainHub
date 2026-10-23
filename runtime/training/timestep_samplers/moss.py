"""Kohya MOSS logit-normal timestep distribution on the 1000-step grid."""
import torch
from .baseline import BaselineTimestepSampler


class MossTimestepSampler(BaselineTimestepSampler):
    def __init__(self, mean=-1.09, std=0.95, schedule_shift=1.0):
        super().__init__()
        self.mean, self.std = float(mean), float(std)
        self.schedule_shift = float(schedule_shift)

    def sample(self, bs, device, *, token_counts=None):
        u = torch.sigmoid(torch.randn(bs, device=device) * self.std + self.mean)
        t = (1000 - (u * 1000).long()).float().div(1000)
        t = self.schedule_shift * t / (1 + (self.schedule_shift - 1) * t)
        return t.clamp(0.001, 0.999)

    def status(self):
        return {"kind": "moss_logit_normal", "mean": self.mean, "std": self.std, "schedule_shift": self.schedule_shift}


def build(args, total_steps):
    return MossTimestepSampler(getattr(args, "moss_timestep_mean", -1.09),
                               getattr(args, "moss_timestep_std", 0.95),
                               getattr(args, "timestep_schedule_shift", 1.0))
