import csv
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DEFAULT_MEDICAL_DOMAINS = ["nih", "guangzhou", "rsna"]
MEDICAL_DATASET_NAME = "medical_cxr"
MEDICAL_NUM_CLASSES = 2

RSNA_CANDIDATE_DIRS = [
    "/kaggle/input/rsna-pneumonia-detection-challenge",
    "/kaggle/input/rsna-pneumonia-detection-challenge-2018",
    "/kaggle/input/rsna-pneumonia-detection",
]


class PerImageZScore:
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
            PerImageZScore(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ]
    )


def get_medical_augment_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            transforms.RandomRotation(degrees=7),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        ]
    )


def try_import_hf_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required for Hugging Face medical dataset loading. "
            "Install it with `pip install datasets`."
        ) from exc
    return load_dataset


def try_import_pydicom():
    try:
        import pydicom  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Reading RSNA DICOM files requires `pydicom`. Install it with `pip install pydicom` "
            "or provide pre-converted PNG/JPG images."
        ) from exc
    return pydicom


@dataclass
class MedicalSample:
    store_id: str
    item_index: int
    label: int
    domain: str


class MedicalStore:
    def __init__(
        self,
        store_id: str,
        domain: str,
        storage_type: str,
        dataset=None,
        image_key: Optional[str] = None,
        records: Optional[List[Dict]] = None,
    ):
        self.store_id = store_id
        self.domain = domain
        self.storage_type = storage_type
        self.dataset = dataset
        self.image_key = image_key
        self.records = records or []

    def get_image(self, item_index: int) -> Image.Image:
        if self.storage_type == "hf":
            example = self.dataset[item_index]
            image = extract_hf_image(example, self.image_key)
            return image.convert("L")
        if self.storage_type == "path":
            record = self.records[item_index]
            return load_path_image(record["path"]).convert("L")
        raise ValueError(f"Unknown storage type: {self.storage_type}")


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


def load_path_image(path: str) -> Image.Image:
    suffix = Path(path).suffix.lower()
    if suffix == ".dcm":
        pydicom = try_import_pydicom()
        dcm = pydicom.dcmread(path)
        array = dcm.pixel_array.astype(np.float32)
        array = array - array.min()
        denom = max(array.max(), 1.0)
        array = (255.0 * (array / denom)).astype(np.uint8)
        return Image.fromarray(array)
    return Image.open(path)


def extract_hf_image(example: Dict, image_key: Optional[str]) -> Image.Image:
    if image_key and image_key in example:
        image = example[image_key]
    else:
        image = None
        for key in ["image", "img", "Image", "pixel_values", "x"]:
            if key in example:
                image = example[key]
                break
    if image is None:
        raise KeyError(f"Could not find an image field in Hugging Face example keys: {list(example.keys())}")

    if isinstance(image, Image.Image):
        return image
    if isinstance(image, dict):
        if "path" in image and image["path"]:
            return Image.open(image["path"])
        if "bytes" in image and image["bytes"]:
            from io import BytesIO

            return Image.open(BytesIO(image["bytes"]))
    raise TypeError(f"Unsupported Hugging Face image payload type: {type(image)}")


def normalize_label_name(label_name: str) -> str:
    return label_name.strip().lower().replace("_", " ").replace("-", " ")


def map_nih_label(example: Dict) -> Optional[int]:
    raw_value = None
    for key in ["Finding Labels", "finding_labels", "labels", "label"]:
        if key in example:
            raw_value = example[key]
            break
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        label_names = [normalize_label_name(v) for v in raw_value.split("|")]
    elif isinstance(raw_value, (list, tuple)):
        label_names = [normalize_label_name(str(v)) for v in raw_value]
    else:
        label_names = [normalize_label_name(str(raw_value))]
    if "pneumonia" in label_names:
        return 1
    if label_names == ["no finding"] or label_names == ["normal"]:
        return 0
    return None


def map_guangzhou_label(example: Dict) -> Optional[int]:
    raw_value = None
    for key in ["label", "labels", "class", "target"]:
        if key in example:
            raw_value = example[key]
            break
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        value = normalize_label_name(raw_value)
        if "pneumonia" in value:
            return 1
        if "normal" in value:
            return 0
        return None
    if isinstance(raw_value, bool):
        return int(raw_value)
    if isinstance(raw_value, (int, np.integer)):
        return int(raw_value)
    return None


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


def summarize_samples(samples: Sequence[MedicalSample]) -> Dict[str, Dict[int, int]]:
    summary: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        summary[sample.domain][sample.label] += 1
    return {domain: dict(counts) for domain, counts in summary.items()}


def discover_rsna_root() -> Optional[str]:
    for candidate in RSNA_CANDIDATE_DIRS:
        if os.path.exists(candidate):
            return candidate
    return None


def load_hf_domain(
    dataset_name: str,
    domain_name: str,
    label_mapper,
    cache_dir: Optional[str] = None,
) -> Tuple[MedicalStore, List[MedicalSample]]:
    load_dataset = try_import_hf_datasets()
    dataset_dict = load_dataset(dataset_name, cache_dir=cache_dir)
    if hasattr(dataset_dict, "keys"):
        if "train" in dataset_dict:
            hf_dataset = dataset_dict["train"]
        else:
            first_split = next(iter(dataset_dict.keys()))
            hf_dataset = dataset_dict[first_split]
    else:
        hf_dataset = dataset_dict

    image_key = None
    if hasattr(hf_dataset, "column_names"):
        for key in ["image", "img", "Image", "pixel_values", "x"]:
            if key in hf_dataset.column_names:
                image_key = key
                break

    store_id = f"{domain_name}_hf"
    store = MedicalStore(
        store_id=store_id,
        domain=domain_name,
        storage_type="hf",
        dataset=hf_dataset,
        image_key=image_key,
    )
    samples: List[MedicalSample] = []
    for item_index, example in enumerate(hf_dataset):
        label = label_mapper(example)
        if label is None:
            continue
        samples.append(
            MedicalSample(
                store_id=store_id,
                item_index=item_index,
                label=int(label),
                domain=domain_name,
            )
        )
    return store, samples


def load_rsna_domain(rsna_root: Optional[str] = None) -> Tuple[Optional[MedicalStore], List[MedicalSample]]:
    rsna_root = rsna_root or discover_rsna_root()
    if not rsna_root:
        return None, []

    label_csv = os.path.join(rsna_root, "stage_2_train_labels.csv")
    class_csv = os.path.join(rsna_root, "stage_2_detailed_class_info.csv")
    image_dir = os.path.join(rsna_root, "stage_2_train_images")
    if not (os.path.exists(label_csv) and os.path.exists(class_csv) and os.path.isdir(image_dir)):
        warnings.warn(f"RSNA root found at {rsna_root}, but expected files are missing. Skipping RSNA domain.")
        return None, []

    class_map: Dict[str, str] = {}
    with open(class_csv, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            class_map[row["patientId"]] = row["class"]

    grouped_target: Dict[str, int] = defaultdict(int)
    with open(label_csv, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped_target[row["patientId"]] = max(grouped_target[row["patientId"]], int(row["Target"]))

    records: List[Dict] = []
    for patient_id, target in grouped_target.items():
        class_name = class_map.get(patient_id, "")
        if class_name == "Lung Opacity":
            label = 1
        elif class_name == "Normal":
            label = 0
        else:
            continue
        dcm_path = os.path.join(image_dir, f"{patient_id}.dcm")
        png_path = os.path.join(image_dir, f"{patient_id}.png")
        jpg_path = os.path.join(image_dir, f"{patient_id}.jpg")
        if os.path.exists(dcm_path):
            image_path = dcm_path
        elif os.path.exists(png_path):
            image_path = png_path
        elif os.path.exists(jpg_path):
            image_path = jpg_path
        else:
            continue
        records.append({"path": image_path, "label": label, "patient_id": patient_id})

    store = MedicalStore(
        store_id="rsna_path",
        domain="rsna",
        storage_type="path",
        records=records,
    )
    samples = [
        MedicalSample(store_id="rsna_path", item_index=index, label=record["label"], domain="rsna")
        for index, record in enumerate(records)
    ]
    return store, samples


def load_medical_domain_bundle(
    domains: Optional[Iterable[str]] = None, cache_dir: Optional[str] = None
) -> Tuple[Dict[str, MedicalStore], Dict[str, List[MedicalSample]], List[str]]:
    requested_domains = list(domains or DEFAULT_MEDICAL_DOMAINS)
    stores: Dict[str, MedicalStore] = {}
    samples_by_domain: Dict[str, List[MedicalSample]] = {}
    unavailable_domains: List[str] = []

    for domain in requested_domains:
        if domain == "nih":
            try:
                store, samples = load_hf_domain(
                    "BahaaEldin0/NIH-Chest-Xray-14",
                    domain_name="nih",
                    label_mapper=map_nih_label,
                    cache_dir=cache_dir,
                )
            except Exception as exc:  # pragma: no cover - depends on environment
                warnings.warn(f"Failed to load NIH domain: {exc}")
                unavailable_domains.append(domain)
                continue
            stores[store.store_id] = store
            samples_by_domain["nih"] = samples
        elif domain == "guangzhou":
            try:
                store, samples = load_hf_domain(
                    "hf-vision/chest-xray-pneumonia",
                    domain_name="guangzhou",
                    label_mapper=map_guangzhou_label,
                    cache_dir=cache_dir,
                )
            except Exception as exc:  # pragma: no cover - depends on environment
                warnings.warn(f"Failed to load Guangzhou domain: {exc}")
                unavailable_domains.append(domain)
                continue
            stores[store.store_id] = store
            samples_by_domain["guangzhou"] = samples
        elif domain == "rsna":
            try:
                store, samples = load_rsna_domain()
            except Exception as exc:  # pragma: no cover - depends on environment
                warnings.warn(f"Failed to load RSNA domain: {exc}")
                unavailable_domains.append(domain)
                continue
            if store is None or not samples:
                unavailable_domains.append(domain)
                continue
            stores[store.store_id] = store
            samples_by_domain["rsna"] = samples
        else:
            unavailable_domains.append(domain)

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
            client_domain_summary[client_id] = summarize_samples(samples)
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
