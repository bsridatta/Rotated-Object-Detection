import torch
import torch.nn as nn
from src.models.mish import Mish
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Similar to the building block in ResNet https://arxiv.org/abs/1512.03385
    2*conv+bn layers with residual connection. 
    Represents each 'stage' from which feature pyramid is build  

    Arguments:
        i_channels {int} -- input channels to the block
        o_channels {int} -- ouput channels from the block

    Keyword Arguments:
        stride {int} -- replace pooling with stride (default: {2})
        padding {int} -- preserve feature map dims (default: {1})
    """

    def __init__(self, i_channels, o_channels, stride=2, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(i_channels, o_channels,
                               kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=o_channels)
        self.activ = Mish()
        self.conv2 = nn.Conv2d(o_channels, o_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=o_channels)

        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(i_channels, o_channels, kernel_size=1,
                          stride=2, bias=False),
                nn.BatchNorm2d(o_channels)
            )

    def forward(self, x):
        residual = x
        x = self.bn1(self.conv1(x))
        x = self.activ(x)
        x = self.bn2(self.conv2(x))

        # downsample residual to match conv output
        if self.downsample is not None:
            residual = self.downsample(residual)

        x = x + residual
        x = self.activ(x)
        return x


class Detector_FPN(nn.Module):
    """ResNet(18) inspired architecture with Feature Pyramid Network
    Classification from the top of the pyramid and reg. from bottom

    References: 
        FPN for Object Detection - https://arxiv.org/pdf/1612.03144.pdf
    Code References:
        https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        https://github.com/kuangliu/pytorch-fpn/blob/master/fpn.py
        https://keras.io/examples/vision/retinanet/
    """

    def __init__(self):
        super(Detector_FPN, self).__init__()
        self.image_size = 200
        self.activ = Mish()

        # output filters at each 'stage'
        filters = [16, 32, 64, 128, 256]

        # Bottom-Up pathway
        # Extremely important to not have bigger stride in the top layers
        # intuition is to have precise information of the ship vertices
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.Conv2d(8, filters[0], kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.MaxPool2d(2),
        )

        # Stages used to build pyramid
        self.conv_c2 = ConvBlock(filters[0], filters[1], stride=2, padding=1)
        self.conv_c3 = ConvBlock(filters[1], filters[2], stride=2, padding=1)
        self.conv_c4 = ConvBlock(filters[2], filters[3], stride=2, padding=1)
        self.conv_c5 = ConvBlock(filters[3], filters[4], stride=2, padding=1)

        # pyramid channels fixed to 256 - as mentioned in the cited paper
        py_chs = 256

        # Top-Down pathway
        # Rest of the pyramid is built by upsampling from pyramid top
        self.conv_pyramid_top = nn.Conv2d(filters[4], py_chs, 1)

        # Lateral Connections
        # reduce bottom up channels to match pyramid
        self.conv_c2_red = nn.Conv2d(filters[1], py_chs, kernel_size=1)
        self.conv_c3_red = nn.Conv2d(filters[2], py_chs, kernel_size=1)
        self.conv_c4_red = nn.Conv2d(filters[3], py_chs, kernel_size=1)

        # smooth pyramid levels to reduce aliasing effect from upsampling
        self.conv_p2_smooth = nn.Conv2d(
            py_chs, py_chs, kernel_size=3, padding=1)
        self.conv_p3_smooth = nn.Conv2d(
            py_chs, py_chs, kernel_size=3, padding=1)
        self.conv_p4_smooth = nn.Conv2d(
            py_chs, py_chs, kernel_size=3, padding=1)

        # average pooling to flatten features for cls. and reg. heads
        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # classification and regression subnets
        self.classifier = nn.Sequential(
            # conserve params
            # nn.Conv2d(py_chs, 1,
            #           kernel_size=3, stride=2, bias=False),
            nn.Flatten(),
            # nn.Linear(py_chs, py_chs),
            # self.activ,
            nn.Linear(py_chs, 1),
            # nn.Sigmoid() ## using BCE with logits
        )

        self.regressor = nn.Sequential(
            # conserve params
            # nn.Conv2d(py_chs, 5,
            #           kernel_size=3, stride=2, bias=False),
            nn.Flatten(),
            # nn.Linear(py_chs, py_chs),
            # self.activ,
            nn.Linear(py_chs, 5)
        )

    def forward(self, x):

        # Bottom-Up pathway
        c1 = self.conv_c1(x)
        c2 = self.conv_c2(c1)
        c3 = self.conv_c3(c2)
        c4 = self.conv_c4(c3)
        c5 = self.conv_c5(c4)

        # Top-Down pathway
        p5 = self.conv_pyramid_top(c5)
        # add lateral connections from reduced bottom up to inner pyramid
        p4 = self._upsample_add(p5, self.conv_c4_red(c4))
        p3 = self._upsample_add(p4, self.conv_c3_red(c3))
        p2 = self._upsample_add(p3, self.conv_c2_red(c2))

        # smoothing the pyramid
        # p5 only goes through 1x1 so no need to smooth
        # conserve parameters with assumption that finer res. can preditict better boxes
        # else get highest conf. pred or smthg more complex
        # p4 = self.conv_p4_smooth(p4)
        # p3 = self.conv_p3_smooth(p3)
        p2 = self.conv_p2_smooth(p2)

        # top of the pyramid has the best semantic features
        # and the lowest or the finest layer has best global features
        cls_feat = self.avg_pooling(p5)
        reg_feat = self.avg_pooling(p2)

        classification = self.classifier(cls_feat)
        regression = self.regressor(reg_feat)

        p_ship = classification.view(x.shape[0], 1)
        bbox = regression.view(x.shape[0], 5)

        return torch.cat((p_ship, bbox), dim=1)

    def _upsample_add(self, p_prev, lc):
        """takes a pyramid layer, upsamples by factor of 2 and adds corres. lateral connections

        Arguments:
            p_prev {tensor} -- coarser feature map 
            lc {tensor} -- lateral connection

        Returns:
            finer feature map, lower pyramid layer
        """
        p = F.interpolate(p_prev, size=(lc.shape[-2:]), mode='nearest')
        return p+lc


# Run file to see summary
if __name__ == "__main__":
    from torchsummary import summary

    inp = torch.rand((2, 1, 200, 200))
    net = Detector_FPN()
    out = net(inp)

    # print(out.shape)
    summary(net, inp.shape[1:])
    # print(net)
