# Implementing the ResNet 50 Architecture from scratch with help of PyTorch. 

# Importing the required Libraries. 
import torch 
import torch.nn as nn 
import torch.nn.functional as F

# Custom ReLU Activation function. 
# f(x) = max(0, x)
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
        self.in_channels = 64 