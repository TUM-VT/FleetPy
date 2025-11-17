import torch
from typing import Optional


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        """
        Implementation of Focal Loss with alpha balancing.
        Args:
            alpha: Weighting factor for the rare class (default 0.25)
            gamma: Focusing parameter that adjusts the rate at which easy examples are down-weighted (default 2.0)
            pos_weight: Optional tensor of positive weights for balancing
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.eps = 1e-7

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss with alpha balancing.

        Args:
            inputs: Raw logits from the model
            targets: Binary target values (0 or 1)

        Returns:
            Focal loss value
        """
        # Get probabilities with numerical stability
        probs = torch.sigmoid(inputs)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)

        # Compute p_t (probability for target class)
        p_t = targets * probs + (1 - targets) * (1 - probs)

        # Compute alpha_t (alpha weight for target class)
        if self.pos_weight is not None:
            alpha_t = targets * self.pos_weight + (1 - targets)
        else:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Compute modulating factor
        focal_weight = (1 - p_t).pow(self.gamma)

        # Compute CE loss
        ce_loss = -torch.log(p_t)

        # Combine all terms
        focal_loss = alpha_t * focal_weight * ce_loss

        return focal_loss.mean()
