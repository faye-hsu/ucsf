import segmentation_models_pytorch as smp


def build_model(num_classes=3, encoder_name="mobilenet_v2"):
    """Lightweight U-Net suitable for CPU training/inference.

    mobilenet_v2 encoder keeps parameter count and FLOPs low; swap to
    resnet18/34 if you later get GPU access and want more capacity.
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=1,
        classes=num_classes,
    )
