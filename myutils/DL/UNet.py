import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Iterable, Tuple, List

class CustModule(nn.Module):
    def __init__(self, type: str = '2D'):
        super().__init__()
        self.layers = nn.Identity()
        self.Conv = nn.Conv2d if type == '2D' else nn.Conv3d
        self.BatchNorm = nn.BatchNorm2d if type == '2D' else nn.BatchNorm3d
        self.MaxPool = nn.MaxPool2d if type == '2D' else nn.MaxPool3d
        self.ConvTranspose = nn.ConvTranspose2d if type == '2D' else nn.ConvTranspose3d

    def pool_down_kernel(self, depth_down: bool = True):
        if depth_down:
            return 2
        else:
            return 1, 2, 2

    def convTransParam(self, type = '2D', depth_up = True):
        if depth_up:
            return {'kernel_size': 2, 'stride': 2}
        else:
            return {'kernel_size': (1, 2, 2), 'stride': (1, 2, 2)}

    def forward(self, x):
        return self.layers(x)


class ConvLayer(CustModule):
    def __init__(self, in_c: int, out_c: int, type: int = '2D'):
        super().__init__(type)
        self.layers = nn.Sequential(
            self.Conv(in_c, out_c, kernel_size = 3,  stride=1, padding= 1),
            self.BatchNorm(out_c),
            nn.ReLU(inplace = True),
            self.Conv(out_c, out_c, kernel_size=3, stride=1, padding=1),
            self.BatchNorm(out_c),
            nn.ReLU(inplace = True)
        )


class FinalLayer(CustModule):
    def __init__(self, channels: int, include_sig: bool = True, type: str = '2D'):
        super().__init__(type)
        self.last = nn.Sigmoid() if include_sig else nn.Identity()
        self.layers = nn.ModuleList([])
        for in_c, out_c in zip(channels, channels[1:]):
            if len(self.layers) != 0:
                self.layers.append(nn.ReLU())
            self.layers.append(self.Conv(in_c, out_c, kernel_size = 1,  stride=1, padding= 0))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.last(x)


class DownLayer(CustModule):
    """
    Downsamples the input by a factor of 2 and processes it with a Conv Layer
    """
    def __init__(self, in_c: int, out_c: int, type: int = '2D', depth_downsample: bool = True):
        super().__init__(type)
        self.layers = nn.Sequential(
            self.MaxPool(kernel_size=self.pool_down_kernel(depth_downsample)),
            ConvLayer(in_c, out_c, type)
        )


class Encoder(CustModule):
    def __init__(self, channels, type = '2D', depth_downsample = None):
        super().__init__(type)
        self.channels = channels
        self.depth_downsample = depth_downsample if (type == '3D' and depth_downsample is not None) else [True] * (len(channels)-2)
        self.layers = nn.ModuleList([ConvLayer(in_c= self.channels[0], out_c=self.channels[1], type = type)])
        # Shift channels by one and zip to generate down-sampling path
        for in_c, out_c, depth_down in zip(self.channels[1:], self.channels[2:], self.depth_downsample):
            self.layers.append(DownLayer(in_c, out_c, type, depth_down))

    def forward(self, x):
        features = []
        for l in self.layers:
            x = l(x)
            features.append(x)
        return features


class UpLayer(CustModule):
    """
    First upsamples the input by a factor of 2, concats the skip-connection and outputs it through another Conv-Layer
    """
    def __init__(self, in_c: int, concat_c: int,  out_c: int, type: str = '2D', 
                 depth_upsample: bool = True, interpolate: bool = True):
        super().__init__(type)
        if interpolate:
            self.upsample = partial(F.interpolate, scale_factor=2, mode="nearest")
            self.layers = nn.Sequential(
                nn.Dropout(p = 0.25),
                ConvLayer(in_c + concat_c, out_c, type)
            )
        else:
            self.upsample = self.ConvTranspose(in_c, out_c, **self.convTransParam(type, depth_upsample))
            self.layers = nn.Sequential(
                nn.Dropout(p = 0.25),
                ConvLayer(out_c + concat_c, out_c, type)
            )

    def forward(self, x, skip = None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim = 1)
        return self.layers(x)


class Decoder(CustModule):
    def __init__(self, channels: Iterable[int], enc_channels: Iterable[int], 
                 type: str = '2D', depth_upsample: Optional[Iterable[int]] = None, interpolate = False) -> None:
        super().__init__(type)
        assert channels[0] == enc_channels[-1],\
            "Decoder has to start with the same number of channels as encoder ends"
        self.channels = channels
        self.enc_channels = enc_channels[-2:0:-1]  # Reverse and exclude the first entry and last
        self.depth_upsample = depth_upsample[::-1] if (type == '3D' and depth_upsample is not None) \
                                                   else [True] * (len(channels) - 1)

        self.layers = nn.ModuleList([])
        for (in_c, enc_c, out_c, d_upsample) in zip(self.channels, self.enc_channels, self.channels[1:], self.depth_upsample):
            self.layers.append(UpLayer(in_c=in_c,
                                       concat_c=enc_c,
                                       out_c=out_c, 
                                       type=type, 
                                       depth_upsample=d_upsample, 
                                       interpolate=interpolate))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Go backwards through the features and make use of the UpLayers which first upsample and then
        concatenate the skip-connections before another convolution
        """
        x = features[-1]
        for layer, feat in zip(self.layers, features[-2::-1]):
            x = layer(x, feat)
        return x


class UNet(nn.Module):
    """
    Decoder- and Encoderchannels should have the same length
    Decoder has to start with the same number of channels as encoder ends
    """
    def __init__(self, encoder_channels: Iterable[int], decoder_channels: Iterable[int], type: str = '3D',
                 depth_downsample: Optional[Iterable[int]] = None, interpolate: bool = False,
                 device: Optional[str] = None) -> None:
        super().__init__()
        self.depth_downsampling = depth_downsample

        # Check input
        self._check_args(encoder_channels, decoder_channels, type, depth_downsample)

        # Build model
        self.output_dim = decoder_channels[-1]
        self.encoder = Encoder(encoder_channels, type, depth_downsample)
        self.decoder = Decoder(decoder_channels[:(len(encoder_channels)-1)], encoder_channels, type, depth_downsample, interpolate)
        # Use the layers not used in the U-architecture for the final layers
        self.final = FinalLayer(channels =decoder_channels[(len(encoder_channels)-2):], include_sig=False, type = type)

        # Set device
        self.device = device
        if device is not None:
            self.to(self.device)

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features)
        return self.final(features)

    def comp_features(self, x):
        features = self.encoder(x)
        features = self.decoder(features)
        return features

    def predict(self, x):
        res = self.forward(x)
        return torch.argmax(res, dim = 1)

    def _check_args(self, encoder_channels, decoder_channels, type, depth_downsample):
        assert len(encoder_channels) <= len(decoder_channels), \
            'Decoder needs to be longer than encoder'
        assert decoder_channels[0] == encoder_channels[-1],\
            f"Decoder has to start with the same number of channels as encoder ends: {decoder_channels[0]} vs {encoder_channels[-1]}"

        assert type in ['2D', '3D'], "Type has to be either 2D or 3D"
        if type == '2D':
            assert depth_downsample is None, "If type is 2D, there is no depth downsampling possibility!"
        if depth_downsample is not None:
            assert len(depth_downsample) == len(encoder_channels) - 2
        

if __name__ == '__main__':
    from myutils.DL.Debugging import VerboseExecution

    class VerboseExecution(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model

            for name, layer in self.model.layers.named_children():
                layer.__name__ = name
                layer.register_forward_hook(
                    lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)
            

    encoder_channels=[1, 64, 128, 256, 512]
    decoder_channels=[512, 256, 128, 64, 1]
    unet = UNet(encoder_channels, decoder_channels, type='2D')


    x = torch.randn((4, 1, 64, 64))
    
    features = VerboseExecution(unet.encoder)(x)







