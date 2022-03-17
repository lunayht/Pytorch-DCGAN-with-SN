import torch.nn as nn

from spectral_norm import SpectralNorm

NUM_OF_CHANNELS = 3
Z_SIZE = 100
GEN_FEATURE_MAP_SIZE = 64
DISC_FEATURE_MAP_SIZE = 64


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(Z_SIZE, GEN_FEATURE_MAP_SIZE * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(GEN_FEATURE_MAP_SIZE * 8),
            nn.ReLU(True),
            # state size. (GEN_FEATURE_MAP_SIZE*8) x 4 x 4
            nn.ConvTranspose2d(
                GEN_FEATURE_MAP_SIZE * 8, GEN_FEATURE_MAP_SIZE * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(GEN_FEATURE_MAP_SIZE * 4),
            nn.ReLU(True),
            # state size. (GEN_FEATURE_MAP_SIZE*4) x 8 x 8
            nn.ConvTranspose2d(
                GEN_FEATURE_MAP_SIZE * 4, GEN_FEATURE_MAP_SIZE * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(GEN_FEATURE_MAP_SIZE * 2),
            nn.ReLU(True),
            # state size. (GEN_FEATURE_MAP_SIZE*2) x 16 x 16
            nn.ConvTranspose2d(
                GEN_FEATURE_MAP_SIZE * 2, GEN_FEATURE_MAP_SIZE, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(GEN_FEATURE_MAP_SIZE),
            nn.ReLU(True),
            # state size. (GEN_FEATURE_MAP_SIZE) x 32 x 32
            nn.ConvTranspose2d(
                GEN_FEATURE_MAP_SIZE, NUM_OF_CHANNELS, 4, 2, 1, bias=False
            ),
            nn.Tanh()
            # state size. (NUM_OF_CHANNELS) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (NUM_OF_CHANNELS) x 64 x 64
            SpectralNorm(
                nn.Conv2d(NUM_OF_CHANNELS, GEN_FEATURE_MAP_SIZE, 4, 2, 1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (GEN_FEATURE_MAP_SIZE) x 32 x 32
            SpectralNorm(
                nn.Conv2d(
                    GEN_FEATURE_MAP_SIZE, GEN_FEATURE_MAP_SIZE * 2, 4, 2, 1, bias=False
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (GEN_FEATURE_MAP_SIZE*2) x 16 x 16
            SpectralNorm(
                nn.Conv2d(
                    GEN_FEATURE_MAP_SIZE * 2,
                    GEN_FEATURE_MAP_SIZE * 4,
                    4,
                    2,
                    1,
                    bias=False,
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (GEN_FEATURE_MAP_SIZE*4) x 8 x 8
            SpectralNorm(
                nn.Conv2d(
                    GEN_FEATURE_MAP_SIZE * 4,
                    GEN_FEATURE_MAP_SIZE * 8,
                    4,
                    2,
                    1,
                    bias=False,
                )
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (GEN_FEATURE_MAP_SIZE*8) x 4 x 4
            SpectralNorm(nn.Conv2d(GEN_FEATURE_MAP_SIZE * 8, 1, 4, 1, 0, bias=False)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x)
