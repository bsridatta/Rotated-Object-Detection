import torch
import torch.nn as nn


class Baseline(nn.Module):
    """
    PyTorch equivalent of the given baseline model
    """

    def __init__(self):
        super(Baseline, self).__init__()
        self.image_size = 200
        self.n_filters = [x*8 for x in [1, 2, 4, 8, 16, 32, 64]]
        self.features = self._build_features(self.n_filters)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_filters[-1], 5))

    def _build_features(self, n_filter):
        """Generate feature/backbone network

        Arguments:
            n_filter {list} -- number of filter for each conv block

        Returns:
            feature extraction module 
        """
        layers = nn.ModuleList()

        i_channels = 1
        for i in n_filter:
            o_channels = i
            layers.append(nn.Conv2d(i_channels, o_channels,
                                    kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_features=o_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            i_channels = o_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Run file to see summary
if __name__ == "__main__":
    from torchsummary import summary

    inp = torch.rand((2, 1, 200, 200))
    net = Baseline()
    out = net(inp)

    summary(net, inp.shape[1:])
    print(out.shape)
