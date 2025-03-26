import torch.nn as nn
import torchvision.models as models

class MyCIFARModel(nn.Module):
    def __init__(self):
        super(MyCIFARModel, self).__init__()
        # 🛠 Model is assigned to self.model — key step for matching state_dict
        self.model = models.resnet18(pretrained=False)  # use pretrained=False here
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.model(x)

