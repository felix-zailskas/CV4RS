import timm


def create_mlp_mixer(channels, num_classes):
    mlp_mixer = timm.create_model(
        "mixer_b16_224.goog_in21k_ft_in1k",
        pretrained=False,
        img_size=120,
        num_classes=num_classes,
        in_chans=channels,
    )
    return mlp_mixer
