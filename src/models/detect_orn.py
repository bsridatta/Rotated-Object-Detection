import torch
import torch.nn as nn
from src.CUDA.ORN.orn.functions import oraligned1d
from src.CUDA.ORN.orn.modules import ORConv2d
from src.models.mish import Mish


class Detector_ORN(nn.Module):
    """Incomplete implementation of Oriented Response Networks, runs only on gpu
    Implicitly learns orientation of objects using ARF(Active Rotation Filters)
    Advatages - better IOU, fewer parameters, faster convergence, should be ideal for the task
    ORN paper - https://arxiv.org/pdf/1701.01833.pdf
    """

    def __init__(self):
        super(Detector_ORN, self).__init__()
        self.image_size = 200

        # Note: filters are not mul by 8
        self.n_filters = [x for x in [1, 2, 4, 8, 16, 32, 64]]

        # self.activ = nn.ReLU()
        self.activ = Mish()
        self.n_orientation = 8

        self.features = self._build_features(
            self.n_filters, self.activ, self.n_orientation
        )

        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(self.n_filters[-1], 1))

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_filters[-1], self.n_filters[-1]),
            self.activ,
            nn.Dropout(),
            nn.Linear(self.n_filters[-1], 5),
        )

    def _build_features(self, n_filter, activ, n_orientation):
        """Generate feature/backbone network

        Arguments:
            n_filter {list} -- number of filter for each conv block
            activ {nn.Module} -- activation function to be used
            n_orientations {int} -- orientations for ARF

        Returns:
            feature extraction module
        """
        layers = nn.ModuleList()
        i_channels = 1
        for i in n_filter:
            o_channels = i

            if i_channels == 1:
                arf_config_ = (1, n_orientation)
            else:
                arf_config_ = n_orientation

            layers.append(
                ORConv2d(
                    i_channels,
                    o_channels,
                    arf_config=arf_config_,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )

            if i != n_filter[-1]:  # mimicing the paper
                layers.append(nn.BatchNorm2d(num_features=o_channels))
                layers.append(activ)
                layers.append(nn.MaxPool2d(2))
            else:
                # last layer of the feature network
                layers.append(activ)

            i_channels = o_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        # orn pooling
        x = oraligned1d(x, self.n_orientation)

        classification = self.classifier(x)
        regression = self.regressor(x)

        return torch.cat((classification, regression), dim=1)


# Run file to see summary
if __name__ == "__main__":
    from torchsummary import summary

    inp = torch.rand((2, 1, 200, 200))
    net = Detector_ORN()
    out = net(inp)

    summary(net, inp.shape[1:])
    print(out.shape)
