from .unet import resnet50_unet, tiny_unet

model_from_name = {
    "resnet50_unet": resnet50_unet,
    "tiny_unet": tiny_unet,
}


def get_model(model_name, n_classes, input_height=416, input_width=608):
    if model_name not in model_from_name:
        raise ValueError("Unknown model name: {}".format(model_name))
    return model_from_name[model_name](n_classes, input_height, input_width)
