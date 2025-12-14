import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_groups=8):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups, out_channels)
        
        # shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(num_groups, out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out += identity
        out = self.relu(out)

        return out

class ConvNet(nn.Module):
    def __init__(self, input_channels, base_channels=64, final_out_channels = None, num_groups=8):
        super(ConvNet, self).__init__()


        self.final_out_channels = final_out_channels or base_channels * 8

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, base_channels),
            nn.ReLU(inplace=True)
        )

        self.stage1 = ResidualBlock(base_channels, base_channels * 2, stride=2, num_groups=num_groups)
        self.stage2 = ResidualBlock(base_channels * 2, base_channels * 4, stride=2, num_groups=num_groups)
        self.stage3 = ResidualBlock(base_channels * 4, base_channels * 8, stride=2, num_groups=num_groups)

        self.final_conv = nn.Conv2d(base_channels * 8, self.final_out_channels, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.final_conv(x)
        x = self.pool(x)
        return x

# Example usage
if __name__ == "__main__":
    model = ConvNet(input_channels=3, base_channels=64, num_groups=8)
    x = torch.randn(2, 3 * 4, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
