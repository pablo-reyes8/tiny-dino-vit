"""Build Tiny ImageNet datasets, DINO multicrop transforms, and DataLoaders."""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import random_split
from data.data_config import DEFAULT_DATA_CONFIG as DATA_CONFIG

try:
    from datasets import load_dataset
except ModuleNotFoundError:
    load_dataset = None

class DINOTransform:
    def __init__(
        self,
        global_crop_size=64,
        local_crop_size=32,
        num_global_crops=2,
        num_local_crops=4,
        global_crop_scale=(0.5, 1.0),
        local_crop_scale=(0.2, 0.5),
    ):
        for name, scale in (
            ("global_crop_scale", global_crop_scale),
            ("local_crop_scale", local_crop_scale),
        ):
            if len(scale) != 2 or scale[0] > scale[1]:
                raise ValueError(f"{name} must be a (min, max) tuple with min <= max.")

        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops

        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                global_crop_size,
                scale=global_crop_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.15,
                hue=0.05,
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                local_crop_size,
                scale=local_crop_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.15,
                hue=0.05,
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __call__(self, image):
        crops = []

        for _ in range(self.num_global_crops):
            crops.append(self.global_transform(image))

        for _ in range(self.num_local_crops):
            crops.append(self.local_transform(image))

        return crops


def build_eval_transform(image_size=64):
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),),])


class TinyImageNetDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, return_label=False):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.return_label = return_label

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]

        image = sample["image"].convert("RGB")
        label = sample["label"]

        if self.transform is not None:
            image = self.transform(image)

        if self.return_label:
            return image, label

        return image


def seed_worker(worker_id):
    worker_seed = DATA_CONFIG["seed"] + worker_id
    torch.manual_seed(worker_seed)


def build_tinyimagenet_dataloaders(config=DATA_CONFIG):
    """
    Construye datasets y dataloaders para Tiny ImageNet usando DINO multi-crop.

    El train_loader devuelve una lista de crops:
        crops = [
            global_crop_1,  # [B, 3, global_crop_size, global_crop_size]
            global_crop_2,  # [B, 3, global_crop_size, global_crop_size]
            local_crop_1,   # [B, 3, local_crop_size, local_crop_size]
            ...
        ]

    El val_loader devuelve:
        image, label

    porque validación se deja normal para inspección, kNN eval o linear probing.
    """

    torch.manual_seed(config["seed"])

    # --------------------------------------------------------
    # Cargar dataset desde Hugging Face
    # --------------------------------------------------------

    if load_dataset is None:
        raise ModuleNotFoundError(
            "The 'datasets' package is required to load Tiny ImageNet. "
            "Install it with `pip install datasets` or provide a mocked load_dataset."
        )

    hf_dataset = load_dataset(config["dataset_name"])

    print("Available splits:", hf_dataset.keys())

    if "train" not in hf_dataset:
        raise ValueError(
            f"El dataset no tiene split 'train'. Splits disponibles: {list(hf_dataset.keys())}"
        )

    train_hf = hf_dataset["train"]

    # --------------------------------------------------------
    # Construir transforms
    # --------------------------------------------------------

    train_transform = DINOTransform(
        global_crop_size=config["global_crop_size"],
        local_crop_size=config["local_crop_size"],
        num_global_crops=config["num_global_crops"],
        num_local_crops=config["num_local_crops"],
        global_crop_scale=config["global_crop_scale"],
        local_crop_scale=config["local_crop_scale"],
    )

    val_transform = build_eval_transform(
        image_size=config["global_crop_size"]
    )

    # --------------------------------------------------------
    #  Detectar split de validación/test si existe
    # --------------------------------------------------------

    val_split_name = None

    for candidate in ["validation", "valid", "val", "test"]:
        if candidate in hf_dataset:
            val_split_name = candidate
            break

    # --------------------------------------------------------
    # Caso A: existe validation/valid/val/test
    # --------------------------------------------------------

    if val_split_name is not None:
        print(f"Using '{val_split_name}' as validation split.")

        val_hf = hf_dataset[val_split_name]

        train_dataset = TinyImageNetDataset(
            hf_dataset=train_hf,
            transform=train_transform,
            return_label=False,
        )

        val_dataset = TinyImageNetDataset(
            hf_dataset=val_hf,
            transform=val_transform,
            return_label=True,
        )

    # --------------------------------------------------------
    # Caso B: solo existe train, hacemos split manual
    # --------------------------------------------------------

    else:
        print("No validation/test split found. Creating manual train/val split.")

        full_train_dataset = TinyImageNetDataset(
            hf_dataset=train_hf,
            transform=train_transform,
            return_label=False,
        )

        full_val_dataset = TinyImageNetDataset(
            hf_dataset=train_hf,
            transform=val_transform,
            return_label=True,
        )

        val_ratio = config.get("val_ratio", 0.1)

        val_size = int(val_ratio * len(train_hf))
        train_size = len(train_hf) - val_size

        generator = torch.Generator().manual_seed(config["seed"])

        train_indices, val_indices = random_split(
            range(len(train_hf)),
            lengths=[train_size, val_size],
            generator=generator,
        )

        train_dataset = torch.utils.data.Subset(
            full_train_dataset,
            train_indices.indices,
        )

        val_dataset = torch.utils.data.Subset(
            full_val_dataset,
            val_indices.indices,
        )

    # --------------------------------------------------------
    # Crear DataLoaders
    # --------------------------------------------------------

    persistent_workers = (
        config["persistent_workers"]
        if config["num_workers"] > 0
        else False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle_train"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=persistent_workers,
        drop_last=config["drop_last_train"],
        worker_init_fn=seed_worker,)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=persistent_workers,
        drop_last=config["drop_last_val"],
        worker_init_fn=seed_worker,)


    print("Train dataset size:", len(train_dataset))
    print("Val dataset size:", len(val_dataset))
    print("Number of global crops:", config["num_global_crops"])
    print("Number of local crops:", config["num_local_crops"])
    print("Global crop size:", config["global_crop_size"])
    print("Local crop size:", config["local_crop_size"])

    return train_dataset, val_dataset, train_loader, val_loader



