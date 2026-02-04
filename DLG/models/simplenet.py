import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, in_channel=3):
        super(SimpleCNN, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channel, 12, kernel_size=5, padding=2, stride=2),
            nn.Sigmoid(),
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=2),
            nn.Sigmoid(),
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=2),
            nn.Sigmoid(),
        )
        self.fc = nn.Linear(12 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)