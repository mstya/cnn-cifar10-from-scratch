from torch import nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, max_pool_s=2):
        super(CNNBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=max_pool_s)
        )

    def forward(self, x):
        return self.block(x)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout):
        super(SimpleCNN, self).__init__()

        self.conv_block1 = CNNBlock(3, 32)
        self.conv_block2 = CNNBlock(32, 64)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        x = self.classifier(x)

        return x

class SimpleCNNConv3(nn.Module):
    def __init__(self, num_classes, dropout):
        super(SimpleCNNConv3, self).__init__()

        self.conv_block1 = CNNBlock(3, 32)
        self.conv_block2 = CNNBlock(32, 64)
        self.conv_block3 = CNNBlock(64, 128, max_pool_s=1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6272, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout * 0.15),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout * 0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = self.classifier(x)

        return x