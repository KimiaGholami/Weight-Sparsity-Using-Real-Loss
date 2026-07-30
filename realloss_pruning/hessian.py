import torch
import torch.nn as nn


class HessianAccumulator:
    """Accumulates H = (2/N) X^T X for a linear layer's input activations.

    Matches the calibration-Hessian construction shared by SparseGPT and CWS:
    H depends only on the layer's inputs, so a single accumulator per layer is
    shared across all output rows/neurons.
    """

    def __init__(self, layer: nn.Linear):
        self.layer = layer
        d_in = layer.weight.shape[1]
        self.device = layer.weight.device
        self.H = torch.zeros((d_in, d_in), device=self.device, dtype=torch.float64)
        self.n_samples = 0

    def update(self, inp: torch.Tensor) -> None:
        if inp.dim() > 2:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.to(device=self.device, dtype=torch.float64)
        n = inp.shape[0]
        if n == 0:
            return
        self.H.mul_(self.n_samples / (self.n_samples + n))
        self.n_samples += n
        inp = inp * (2.0 / self.n_samples) ** 0.5
        self.H.add_(inp.t() @ inp)

    def make_hook(self):
        def hook(module, inputs):
            self.update(inputs[0])

        return hook

    def get_dampened_hessian(self, damping: float = 0.01) -> torch.Tensor:
        H = self.H.clone()
        diag_idx = torch.arange(H.shape[0], device=H.device)
        dead = H[diag_idx, diag_idx] == 0
        H[dead, dead] = 1.0
        damp = damping * H[diag_idx, diag_idx].mean()
        H[diag_idx, diag_idx] += damp
        return H
