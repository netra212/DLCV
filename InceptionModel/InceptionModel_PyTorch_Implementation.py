# Implementing the Inception model with PyTorch. 
import torch
import torch.nn as nn
import torch.nn.functional as F

# Building an Inception Blocks. 
class InceptionModule(nn.Module):
    def __init__(self, in_channels, f1, f2_r, f2, f3_r, f3, f4):
        super(InceptionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1)

        self.conv3_r = nn.Conv2d(in_channels, f2_r, kernel_size=1)
        self.conv3 = nn.Conv2d(f2_r, f2, kernel_size=3, padding=1)

        self.conv5_r = nn.Conv2d(in_channels, f3_r, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels, f4, kernel_size=1)

        self.pool_proj = nn.Conv2d(in_channels, f4, kernel_size=1)

    def forward(self, x):
        conv1_out = F.relu(self.conv1(x))

        conv3_out = F.relu(self.conv3_r(x))
        conv3_out = F.relu(self.conv3(conv3_out))

        conv5_out = F.relu(self.conv5_r(x))
        conv5_out = F.relu(self.conv5(conv5_out))

        pool_out = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        pool_out = F.relu(self.pool_proj(pool_out))

        return torch.cat([
            conv1_out,
            conv3_out,
            conv5_out,
            pool_out
        ], dim=1)

# Building an Auxillary Classifier. 
class AuxillaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxillaryClassifier, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(128*4*4, 1024) # Assuming input size is 224x224. 
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        x = self.pool(x)
        x = F.relu(self.conv(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Building an GoogleNet. 
class GoogleNet(nn.Module):
    def __init__(self, num_classes=15):
        super(GoogleNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.aux1 = AuxillaryClassifier(512, num_classes)

        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.aux2 = AuxillaryClassifier(528, num_classes)

        self.inception5a = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)

        x = self.inception4a(x)
        aux1_out = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2_out = self.aux2(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        main_out = self.fc(x)

        return main_out, aux1_out, aux2_out

googlenet = GoogleNet()
print(googlenet)