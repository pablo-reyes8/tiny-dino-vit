"""Provide small visualization helpers for inspecting image tensors."""

import matplotlib.pyplot as plt
import torch

def unnormalize_imagenet(x):
    """
    x: Tensor [3, H, W] normalizado con mean/std de ImageNet.
    return: Tensor [3, H, W] en rango aproximado [0, 1].
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)

    x = x * std + mean
    x = x.clamp(0, 1)

    return x

def show_dino_batch_pretty(
    train_loader,
    num_images=6,
    num_global_crops=2,
    figsize_per_image=2.4,
    title="DINO multi-crop batch",
):
    """
    Visualización limpia de un batch DINO multi-crop.

    Estructura esperada del batch:
        batch[0:num_global_crops] -> global crops
        batch[num_global_crops:]  -> local crops

    Cada columna corresponde a una imagen original.
    Cada fila corresponde a un crop distinto.
    """

    batch = next(iter(train_loader))

    num_crops = len(batch)
    num_local_crops = num_crops - num_global_crops
    num_images = min(num_images, batch[0].shape[0])

    row_names = []
    for i in range(num_global_crops):
        row_names.append(f"Global {i + 1}")
    for i in range(num_local_crops):
        row_names.append(f"Local {i + 1}")

    fig, axes = plt.subplots(
        nrows=num_crops,
        ncols=num_images,
        figsize=(figsize_per_image * num_images, figsize_per_image * num_crops),
        squeeze=False)

    fig.suptitle(title, fontsize=18, y=1.01)

    for crop_idx, crop_batch in enumerate(batch):
        for img_idx in range(num_images):
            img = (
                unnormalize_imagenet(crop_batch[img_idx])
                .cpu()
                .permute(1, 2, 0))

            ax = axes[crop_idx, img_idx]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            # Título de columnas solo arriba
            if crop_idx == 0:
                ax.set_title(
                    f"Image {img_idx}",
                    fontsize=12,
                    pad=8)

            if img_idx == 0:
                ax.set_ylabel(
                    row_names[crop_idx],
                    fontsize=12,
                    rotation=0,
                    labelpad=45,
                    va="center",)

            # Borde visual distinto para global/local
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)

                if crop_idx < num_global_crops:
                    spine.set_alpha(0.9)
                else:
                    spine.set_alpha(0.35)

    plt.tight_layout()
    plt.show()
