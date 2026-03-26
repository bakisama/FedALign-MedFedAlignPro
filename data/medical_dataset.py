import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DEFAULT_MEDICAL_DOMAINS = ["aptos", "idrid", "messidor", "messidor2"]
MEDICAL_DATASET_NAME = "medical_dr"
MEDICAL_NUM_CLASSES = 2
POSITIVE_LABEL_NAME = "referable_dr"
NEGATIVE_LABEL_NAME = "non_referable_dr"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
METADATA_SUFFIXES = {".csv", ".xls", ".xlsx"}

DOMAIN_ALIASES = {
    "aptos": ["aptos", "blindness-detection"],
    "idrid": ["idrid"],
    "messidor": ["messidor"],
    "messidor2": ["messidor2", "messidor-2"],
}

IMAGE_COL_ALIASES = [
    "id_code",
    "image_id",
    "image",
    "image name",
    "img",
    "filename",
    "file_name",
    "file",
    "name",
]
LABEL_COL_ALIASES = [
    "diagnosis",
    "dr_grade",
    "retinopathy grade",
    "grade",
    "level",
    "label",
    "class",
    "referable",
    "referable_dr",
    "dr",
]


class PerImageStandardize:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = tensor.mean()
        std = tensor.std()
        if std < 1e-6:
            std = torch.tensor(1.0, dtype=tensor.dtype)
        return (tensor - mean) / std


def get_medical_base_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            PerImageStandardize(),
        ]
    )


def get_medical_augment_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        ]
    )


@dataclass
class MedicalSample:
    store_id: str
    item_index: int
    label: int
    domain: str


class MedicalStore:
    def __init__(self, store_id: str, domain: str, records: Sequence[Dict]):
        self.store_id = store_id
        self.domain = domain
        self.records = list(records)

    def get_image(self, item_index: int) -> Image.Image:
        return Image.open(self.records[item_index]["path"]).convert("RGB")


class MedicalFederatedDataset(Dataset):
    def __init__(
        self,
        stores: Dict[str, MedicalStore],
        samples: Sequence[MedicalSample],
        transform=None,
    ):
        self.stores = stores
        self.samples = list(samples)
        self.transform = transform or get_medical_base_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        store = self.stores[sample.store_id]
        image = store.get_image(sample.item_index)
        image = self.transform(image)
        label = torch.tensor(float(sample.label), dtype=torch.float32)
        return image, label, sample.domain


def normalize_name(value: str) -> str:
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def normalize_referable_dr_label(raw_value) -> Optional[int]:
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return None

    if isinstance(raw_value, (int, np.integer)):
        return int(raw_value >= 2)
    if isinstance(raw_value, (float, np.floating)):
        return int(int(raw_value) >= 2)

    normalized = normalize_name(raw_value)
    if normalized in {"0", "no dr", "normal", "non referable", "non referable dr", "non-referable"}:
        return 0
    if normalized in {
        "1",
        "mild",
    }:
        return 0
    if normalized in {
        "2",
        "3",
        "4",
        "moderate",
        "severe",
        "proliferative",
        "referable",
        "referable dr",
        "rdr",
        "vtdr",
        "positive",
    }:
        return 1
    try:
        return int(int(float(normalized)) >= 2)
    except ValueError:
        return None


def infer_column(columns: Iterable[str], aliases: Sequence[str]) -> Optional[str]:
    normalized_map = {normalize_name(col): col for col in columns}
    for alias in aliases:
        normalized_alias = normalize_name(alias)
        if normalized_alias in normalized_map:
            return normalized_map[normalized_alias]
    return None


def read_metadata_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=";")
    return pd.read_excel(path)


def discover_domain_root(domain: str) -> Optional[Path]:
    env_key = f"{domain.upper()}_ROOT"
    env_value = os.environ.get(env_key)
    if env_value and Path(env_value).exists():
        return Path(env_value)

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        aliases = DOMAIN_ALIASES.get(domain, [domain])
        for child in kaggle_input.iterdir():
            normalized = normalize_name(child.name)
            if any(alias in normalized for alias in aliases):
                return child
    return None


def build_image_index(root: Path) -> Dict[str, Path]:
    image_index: Dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image_index[path.stem.lower()] = path
        image_index[path.name.lower()] = path
        image_index[str(path.relative_to(root)).lower()] = path
    return image_index


def resolve_image_path(image_value, image_index: Dict[str, Path], root: Path) -> Optional[Path]:
    if image_value is None or (isinstance(image_value, float) and np.isnan(image_value)):
        return None
    image_text = str(image_value).strip()
    candidates = [image_text, Path(image_text).stem]
    path_candidate = root / image_text
    if path_candidate.exists():
        return path_candidate
    for suffix in IMAGE_SUFFIXES:
        stem_candidate = root / f"{image_text}{suffix}"
        if stem_candidate.exists():
            return stem_candidate
    for candidate in candidates:
        normalized = candidate.lower()
        if normalized in image_index:
            return image_index[normalized]
        for suffix in IMAGE_SUFFIXES:
            with_suffix = f"{normalized}{suffix}"
            if with_suffix in image_index:
                return image_index[with_suffix]
    return None


def collect_domain_records(domain: str, root: Path) -> List[Dict]:
    image_index = build_image_index(root)
    metadata_paths = [
        path
        for path in root.rglob("*")
        if path.suffix.lower() in METADATA_SUFFIXES and "sample" not in path.name.lower()
    ]
    best_records: List[Dict] = []
    for metadata_path in metadata_paths:
        try:
            df = read_metadata_table(metadata_path)
        except Exception:
            continue
        image_col = infer_column(df.columns, IMAGE_COL_ALIASES)
        label_col = infer_column(df.columns, LABEL_COL_ALIASES)
        if image_col is None or label_col is None:
            continue
        records: List[Dict] = []
        for _, row in df.iterrows():
            label = normalize_referable_dr_label(row[label_col])
            if label is None:
                continue
            image_path = resolve_image_path(row[image_col], image_index, root)
            if image_path is None:
                continue
            records.append({"path": str(image_path), "label": label})
        if len(records) > len(best_records):
            best_records = records
    return best_records


def load_generic_dr_domain(domain: str) -> Tuple[Optional[MedicalStore], List[MedicalSample]]:
    root = discover_domain_root(domain)
    if root is None:
        return None, []

    records = collect_domain_records(domain, root)
    if not records:
        warnings.warn(
            f"Found root for `{domain}` at {root}, but could not build labeled image records. "
            "Set the dataset root explicitly with an environment variable like "
            f"`{domain.upper()}_ROOT` if needed."
        )
        return None, []

    store_id = f"{domain}_path"
    store = MedicalStore(store_id=store_id, domain=domain, records=records)
    samples = [
        MedicalSample(store_id=store_id, item_index=index, label=record["label"], domain=domain)
        for index, record in enumerate(records)
    ]
    return store, samples


def stratified_split(
    samples: Sequence[MedicalSample], val_ratio: float, seed: int
) -> Tuple[List[MedicalSample], List[MedicalSample]]:
    if not samples:
        return [], []
    rng = np.random.default_rng(seed)
    by_label: Dict[int, List[MedicalSample]] = defaultdict(list)
    for sample in samples:
        by_label[sample.label].append(sample)
    train_samples: List[MedicalSample] = []
    val_samples: List[MedicalSample] = []
    for label_samples in by_label.values():
        indices = np.arange(len(label_samples))
        rng.shuffle(indices)
        num_val = int(round(len(indices) * val_ratio))
        if len(indices) > 1:
            num_val = min(max(num_val, 1), len(indices) - 1)
        else:
            num_val = 0
        val_idx = set(indices[:num_val].tolist())
        for idx, sample in enumerate(label_samples):
            if idx in val_idx:
                val_samples.append(sample)
            else:
                train_samples.append(sample)
    return train_samples, val_samples


def split_samples_to_clients(
    samples: Sequence[MedicalSample], num_clients: int, seed: int
) -> List[List[MedicalSample]]:
    rng = np.random.default_rng(seed)
    by_label: Dict[int, List[MedicalSample]] = defaultdict(list)
    for sample in samples:
        by_label[sample.label].append(sample)
    client_samples = [[] for _ in range(num_clients)]
    for label_samples in by_label.values():
        indices = np.arange(len(label_samples))
        rng.shuffle(indices)
        chunks = np.array_split(indices, num_clients)
        for client_id, chunk in enumerate(chunks):
            for idx in chunk:
                client_samples[client_id].append(label_samples[int(idx)])
    return client_samples


def load_medical_domain_bundle(
    domains: Optional[Iterable[str]] = None, cache_dir: Optional[str] = None
) -> Tuple[Dict[str, MedicalStore], Dict[str, List[MedicalSample]], List[str]]:
    del cache_dir  # Unused for the DR path, retained for interface compatibility.
    requested_domains = list(domains or DEFAULT_MEDICAL_DOMAINS)
    stores: Dict[str, MedicalStore] = {}
    samples_by_domain: Dict[str, List[MedicalSample]] = {}
    unavailable_domains: List[str] = []

    for domain in requested_domains:
        try:
            store, samples = load_generic_dr_domain(domain)
        except Exception as exc:  # pragma: no cover - depends on environment
            warnings.warn(f"Failed to load {domain}: {exc}")
            unavailable_domains.append(domain)
            continue
        if store is None or not samples:
            unavailable_domains.append(domain)
            continue
        stores[store.store_id] = store
        samples_by_domain[domain] = samples

    return stores, samples_by_domain, unavailable_domains


def build_medical_federated_splits(
    stores: Dict[str, MedicalStore],
    samples_by_domain: Dict[str, List[MedicalSample]],
    heldout_domain: str,
    num_clients_per_domain: int = 2,
    val_ratio: float = 0.1,
    seed: int = 42,
    image_size: int = 224,
) -> Dict[str, object]:
    available_domains = sorted(samples_by_domain.keys())
    if heldout_domain not in available_domains:
        raise ValueError(f"Held-out domain `{heldout_domain}` is not available. Available domains: {available_domains}")

    source_domains = [domain for domain in available_domains if domain != heldout_domain]
    if not source_domains:
        raise ValueError("At least one source domain is required to build a federated medical split.")

    client_datasets: List[MedicalFederatedDataset] = []
    val_samples: List[MedicalSample] = []
    client_domain_summary: Dict[int, Dict[str, Dict[int, int]]] = {}
    base_transform = get_medical_base_transform(image_size=image_size)

    client_id = 0
    for domain_index, domain in enumerate(source_domains):
        train_samples, domain_val_samples = stratified_split(
            samples_by_domain[domain], val_ratio=val_ratio, seed=seed + domain_index
        )
        val_samples.extend(domain_val_samples)
        partitioned = split_samples_to_clients(
            train_samples, num_clients=num_clients_per_domain, seed=seed + 97 * (domain_index + 1)
        )
        for samples in partitioned:
            if not samples:
                continue
            dataset = MedicalFederatedDataset(stores=stores, samples=samples, transform=base_transform)
            client_datasets.append(dataset)
            client_domain_summary[client_id] = {
                domain: {
                    0: sum(sample.label == 0 for sample in samples),
                    1: sum(sample.label == 1 for sample in samples),
                }
            }
            client_id += 1

    validation_dataset = MedicalFederatedDataset(
        stores=stores,
        samples=val_samples,
        transform=base_transform,
    )
    test_dataset = MedicalFederatedDataset(
        stores=stores,
        samples=samples_by_domain[heldout_domain],
        transform=base_transform,
    )

    return {
        "client_datasets": client_datasets,
        "validation_dataset": validation_dataset,
        "test_dataset": test_dataset,
        "source_domains": source_domains,
        "heldout_domain": heldout_domain,
        "client_domain_summary": client_domain_summary,
        "available_domains": available_domains,
    }
