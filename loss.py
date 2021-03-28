import torch.nn as nn
import torch.nn.functional as F

class CosineContrastiveLoss(nn.Module):
    def __init__(self, margin=0.4):
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cos_sim = F.cosine_similarity(output1, output2)
        loss_cos_con = torch.mean((1-label) * torch.div(torch.pow((1.0-cos_sim), 2), 4) +
                                    (label) * torch.pow(cos_sim * torch.lt(cos_sim, self.margin), 2))
        return loss_cos_con


class BinaryClassficationLoss(nn.Module):
    def __init__(self):
        super(BinaryClassficationLoss, self).__init__()

    def forward(self, output1, output2, label):
        return 1
