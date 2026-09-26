"""One float base per target, three independent traced uint8 RandAugment copies."""
import csv
from pathlib import Path

from common import CLASSES, IDS, ROOT, require
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms.autoaugment import _apply_op
import _klsg_blind

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
NORMALIZE = T.Normalize(MEAN, STD)


class SpeckleNoise:
    def __call__(self, x):
        return (x * (1 + torch.randn_like(x) * 0.1)).clamp_(0, 1)


def base_transform():
    # Same order/parameters as train_cdan_sgd.build_train_transform, without Normalize.
    return T.Compose([T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                      T.RandomHorizontalFlip(0.5), T.RandomRotation(15),
                      T.ColorJitter(brightness=0.15, contrast=0.15), T.ToTensor(), SpeckleNoise()])


def train_transform():
    return T.Compose([base_transform(), NORMALIZE])


def eval_transform():
    return T.Compose([T.Resize((224, 224)), T.ToTensor(), NORMALIZE])


class TracedRandAugment(T.RandAugment):
    """0.24.1 forward with trace of the SAME random draws and operators, not replay sampling."""
    def __init__(self):
        super().__init__(num_ops=3, magnitude=2, num_magnitude_bins=31,
                         interpolation=T.InterpolationMode.NEAREST, fill=0)

    def forward(self, image):
        require(image.device.type == "cpu" and image.dtype == torch.uint8 and image.ndim == 3,
                "RA expects one CPU uint8 image")
        meta = self._augmentation_space(self.num_magnitude_bins, tuple(image.shape[-2:]))
        require(len(meta) == 14, "RA operator pool changed")
        trace = []
        for _ in range(self.num_ops):
            name = list(meta)[int(torch.randint(len(meta), (1,)).item())]
            magnitudes, signed = meta[name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            image = _apply_op(image, name, magnitude, interpolation=self.interpolation, fill=[0.0] * 3)
            trace.append({"op": name, "magnitude": magnitude})
        return image, trace


def target_cluster(base, ra):
    require(base.dtype == torch.float32 and base.shape == (3, 224, 224), "Invalid float base")
    quantized = (base.clamp(0, 1) * 255).round().to(torch.uint8)
    views, traces = [], []
    for _ in range(3):
        view, trace = ra(quantized.clone())
        views.append(NORMALIZE(view.float() / 255))
        traces.append(trace)
    return NORMALIZE(base), torch.stack(views), traces


def source_rows(fold, split="train"):
    require(fold in range(5) and split in ("train", "test"), "Invalid source fold/split")
    rows = []
    with open(ROOT / "data/split/splits.csv", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["domain"] != "arm2" or int(row["fold"]) != fold or row["split"] != split:
                continue
            rows.append((str((ROOT / row["filepath"]).resolve()), CLASSES.index(row["label"]), row["filepath"]))
    expected = [72, 320] if split == "train" else [18, 80]
    require([sum(r[1] == i for r in rows) for i in range(2)] == expected, "Source split identity/count changed")
    require(len({r[2] for r in rows}) == len(rows), "Duplicate source IDs")
    return rows


class SourceDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform
        self.images = []
        for path, _, _ in rows:
            with Image.open(path) as im:
                self.images.append(im.convert("RGB").copy())

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.rows[i][1], self.rows[i][2]


class BlindTargetDataset(Dataset):
    """No task-label field: sample_id is the immutable manifest row number."""
    def __init__(self, training):
        self.training = training
        self.images = []
        items = _klsg_blind.read_klsg_unlabeled(ROOT)
        require(len({path for path, _ in items}) == len(IDS), "Duplicate target manifest paths")
        for path, forbidden in items:
            require(forbidden is _klsg_blind.FORBIDDEN_LABEL, "Unblinded target item")
            with Image.open(path) as im:
                self.images.append(im.convert("RGB").copy())
        self.transform = base_transform() if training else eval_transform()
        self.ra = TracedRandAugment() if training else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        if not self.training:
            return self.transform(self.images[i]), i
        original, views, traces = target_cluster(self.transform(self.images[i]), self.ra)
        return {"original": original, "views": views, "id": i, "operations": traces}


def collate_target(batch):
    return {"original": torch.stack([x["original"] for x in batch]),
            "views": torch.stack([x["views"] for x in batch]),
            "ids": torch.tensor([x["id"] for x in batch], dtype=torch.int64),
            "operations": [x["operations"] for x in batch]}


def training_loaders(fold):
    rows = source_rows(fold)
    source = DataLoader(SourceDataset(rows, train_transform()), batch_size=64,
                        shuffle=True, num_workers=0, drop_last=True)
    target = DataLoader(BlindTargetDataset(training=True), batch_size=64, shuffle=True,
                        num_workers=0, drop_last=True, collate_fn=collate_target)
    require(len(source) == 6 and len(target) == 8, "Loader schedule changed")
    return source, target, [r[2] for r in rows]
