import torch
import torch.nn as nn

from .mish import Mish


class Detector(nn.Module):
    """Equivalent to baseline architecture with addition of
    classification and regression heads to output 6 attr. p_ship,x,y,yaw,h,w
    """

    def __init__(self):
        super(Detector, self).__init__()
        self.image_size = 200
        self.n_filters = [x * 8 for x in [1, 2, 4, 8, 16, 32, 64]]

        # self.activ = nn.ReLU()
        self.activ = Mish()

        self.features = self._build_features(self.n_filters, self.activ)

        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(self.n_filters[-1], 1))

        self.regressor = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(self.n_filters[-1], self.n_filters[-1]),
            # self.activ,
            # nn.Dropout(),
            nn.Linear(self.n_filters[-1], 5),
        )

    def _build_features(self, n_filter, activ):
        """Generate feature/backbone network

        Arguments:
            n_filter {list} -- number of filter for each conv block
            activ {nn.Module} -- activation function to be used

        Returns:
            feature extraction module
        """
        layers = nn.ModuleList()

        i_channels = 1
        for i in n_filter:
            o_channels = i

            layers.append(
                nn.Conv2d(
                    i_channels,
                    o_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(num_features=o_channels))
            layers.append(activ)
            layers.append(nn.MaxPool2d(2))

            i_channels = o_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        classification = self.classifier(x)
        regression = self.regressor(x)

        return torch.cat((classification, regression), dim=1)


# Run file to see summary
if __name__ == "__main__":
    from torchsummary import summary

    inp = torch.rand((2, 1, 200, 200))
    net = Detector()
    out = net(inp)

    summary(net, inp.shape[1:])
    print(out.shape)
