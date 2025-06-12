import torch
import torch.nn as nn

class MyCustomLearnableLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss1 = nn.CrossEntropyLoss()
        self.loss2 = nn.CrossEntropyLoss()
        self.loss3 = nn.CrossEntropyLoss()

        # log(sigma^2) for each loss (learnable parameters)
        self.log_sigma1 = nn.Parameter(torch.tensor(0.0))
        self.log_sigma2 = nn.Parameter(torch.tensor(0.0))
        self.log_sigma3 = nn.Parameter(torch.tensor(0.0))

    def forward(self, logits1, logits2, logits3, target):
        loss1 = self.loss1(logits1, target)
        loss2 = self.loss2(logits2, target)
        loss3 = self.loss3(logits3, target)

        # Apply uncertainty weighting formula
        total_loss = (
            torch.exp(-self.log_sigma1) * loss1 + self.log_sigma1 +
            torch.exp(-self.log_sigma2) * loss2 + self.log_sigma2 +
            torch.exp(-self.log_sigma3) * loss3 + self.log_sigma3
        ) * 0.5  # mỗi loss có 1/2 * ...

        return {
            'loss': total_loss,
            'loss_naim': loss1,
            'loss_vitbi': loss2,
            'loss_combined': loss3,
            'sigma1': torch.exp(self.log_sigma1).detach(),
            'sigma2': torch.exp(self.log_sigma2).detach(),
            'sigma3': torch.exp(self.log_sigma3).detach(),
        }
    

class MyCustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss1 = nn.CrossEntropyLoss()
        self.loss2 = nn.CrossEntropyLoss()
        self.loss3 = nn.CrossEntropyLoss()

    def forward(self, logits1, logits2, logits3, target):
        loss1 = self.loss1(logits1, target)
        loss3 = self.loss3(logits3, target)

        if logits2 is not None:
            loss2 = self.loss2(logits2, target)
            total_loss = loss1 + loss2 + loss3
        else:
            loss2 = torch.tensor(0.0, device=loss1.device)
            total_loss = loss1 + loss3

        return {
            'loss': total_loss,
            'loss_naim': loss1,
            'loss_vitbi': loss2,
            'loss_combined': loss3
        }

