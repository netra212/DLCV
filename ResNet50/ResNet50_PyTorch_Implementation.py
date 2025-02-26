# Implementing the ResNet 50 Architecture from scratch with help of PyTorch. 

# Importing the required Libraries. 
import torch 
import torch.nn as nn 
import torch.nn.functional as F

# Custom ReLU Activation function. 
# f(x) = max(0, x)

'''
# Why Use ReLU ?
- Prevents Vanishing Gradient - Unlike Sigmoid or tanh, it does not squash values between -1 & 1. 
- Computationally Efficient - Simple max operation, making it fast. 
- Better for Deep Networks - Helps with Learning Complex patterns. 

# Limitation:
 - Dying ReLU Problem - If neurons output only 0, they stop learning. This is why variants like Leaky ReLU exist.
'''
def my_relu(x):
    # if x > 0:
    #     return x 
    # else:
    #    return torch.zeros_like(x)
    
    return torch.where(x>0, x, torch.zeros_like(x))

# Building an Residual block. 
class BottleNeck(nn.Module):
    expansion = 4 # Output channels will be 4 times the input channels. 

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleNeck, self).__init__()

        # 1x1 Convolution (Reducing the dimensions.)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) # out_channels -> featured map after 1st convolution. 

        # 3x3 Convolution (Processing).
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 Convolution (Restore dimension). 
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample
    
    def forward(self, x):
        identity = x # saving input as identity shortcut. 

        out = self.conv1(x)
        out = self.bn1(out)
        out = my_relu(out) # implmementing the custom relu function. 

        out = self.conv2(out)
        out = self.bn2(out)
        out = my_relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # If the dimension is changed. 
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity # Adding the shortcut connection before an activation function. 
        out = my_relu(out) 


# Building an ResNet50 Architecture. 
class ResNet50(nn.Module):

    def __init__(self, num_classes=15):
        super(ResNet50, self).__init__()

        self.in_channels = 64  # Initial input channels. 

        # Initial Convolution Layer. 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = my_relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks (BottleNeck Layers).
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        # Global Average Pooling & Fully Connected Layer. 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)
    
    def _make_layer(self, out_channels, blocks, stride):
        downsample=None
        if stride != 1 or self.in_channels != out_channels * BottleNeck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
        
        layers = []
        layers.append(BottleNeck(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BottleNeck.expansion # Update the input channels. 

        for _ in range(1, blocks): # block =3, 4, 6, 3
            layers.append(BottleNeck(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Instantiate the Model. 
model = ResNet50(num_classes=15)
print(model)