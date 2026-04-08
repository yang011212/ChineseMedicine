import torch
import torch.nn as nn
import torch.nn.functional as F


def get_segmentation_model(encoder, decoder, n_classes, input_height, input_width):
    class SegmentationModel(nn.Module):
        def __init__(self, encoder, decoder, n_classes, input_height, input_width):
            super(SegmentationModel, self).__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.n_classes = n_classes
            self.input_height = input_height
            self.input_width = input_width
            self.output_height = input_height
            self.output_width = input_width
            self.model_name = ""

        def forward(self, x):
            features = self.encoder(x)
            output = self.decoder(features)
            if output.size()[2:] != (self.input_height, self.input_width):
                output = F.interpolate(
                    output,
                    size=(self.input_height, self.input_width),
                    mode="bilinear",
                    align_corners=True,
                )
            return output

    model = SegmentationModel(encoder, decoder, n_classes, input_height, input_width)
    return model


def reshape_for_softmax(x):
    batch_size, channels, height, width = x.size()
    x = x.permute(0, 2, 3, 1).contiguous()
    x = x.view(batch_size, height * width, channels)
    return x


def apply_softmax(x):
    return F.softmax(x, dim=-1)
