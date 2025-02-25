"""
This module contains generators that generate spike event from probability space
"""


from typing import Optional, Tuple

import torch
from torch import nn

from ..functional import HardDiffArgmax, SoftDiffArgmax


class GumbelSoftmaxCustom(nn.Module):
    """
    Custom implementation of the Gumbel Softmax function.

    Args:
        tau (float): The temperature parameter for controlling the softmax distribution. Default is 20.0.
        sample_num (int): The number of samples to generate. Default is 1.
        use_soft (bool): Whether to use soft argmax or hard argmax. Default is True.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]: A tuple containing the hard event, soft event (if use_soft is True), and indices.
    """

    def __init__(
        self,
        tau: float = 20.0,
        sample_num: int = 1,
        use_soft: bool = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.sample_num = sample_num
        self.use_soft = use_soft
        self.hard_argmax = HardDiffArgmax()
        self.soft_argmax = SoftDiffArgmax()

    def forward(
        self,
        alpha: torch.Tensor,
        indices: torch.Tensor,
        use_log: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the Gumbel Softmax function.

        Args:
            alpha (torch.Tensor): The input tensor.
            indices (torch.Tensor): The indices tensor.
            use_log (bool): Whether to use logarithm or not. Default is True.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]: A tuple containing the hard event, soft event (if use_soft is True), and indices.
        """
        # sample Gumbel variable
        sample_nnz: torch.Tensor = torch.rand(
            (self.sample_num, alpha.shape[0], 3), device=alpha.device
        )
        _log_log_u = -torch.log(-torch.log(sample_nnz + 1e-8))

        indices = indices.unsqueeze(0).repeat_interleave(self.sample_num, dim=0)
        if use_log:
            alpha_plus_u = (
                torch.log(
                    alpha.unsqueeze(0).repeat_interleave(self.sample_num, dim=0) + 1e-10
                )
                + _log_log_u
            )
        else:
            alpha_plus_u = (
                alpha.unsqueeze(0).repeat_interleave(self.sample_num, dim=0)
                + _log_log_u
            )
        sampled_probability_3d = torch.softmax(alpha_plus_u / self.tau, dim=-1)

        hard_event: torch.Tensor = self.hard_argmax.apply(sampled_probability_3d)  # type: ignore
        if self.use_soft:
            soft_event = self.soft_argmax(sampled_probability_3d)  # type: ignore
            hard_event.detach_()
        else:
            soft_event = None

        return hard_event, soft_event, indices
