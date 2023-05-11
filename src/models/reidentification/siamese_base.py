from torch import nn
import torch.nn.functional as F


class SiameseBasicCNN(nn.Module):
    """Простая сиамская сверточная нейронная сеть"""

    def __init__(self) -> None:
        super(SiameseBasicCNN, self).__init__()
        self.name = 'basic_cnn'
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64,
                      kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(
                1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64, 256),
            nn.Tanh(),
            nn.Linear(256, 2),
        )

    def forward(self, x1, x2):
        output1 = self.cnn1(x1)
        output1 = output1.view(output1.size()[0], -1)
        output1 = self.fc1(output1)
        output2 = self.cnn1(x2)
        output2 = output2.view(output2.size()[0], -1)
        output2 = self.fc1(output2)

        return F.pairwise_distance(
            output1, output2, keepdim=True)
