from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.medical_dataset import (
    MEDICAL_DATASET_NAME,
    MEDICAL_NUM_CLASSES,
    get_medical_augment_transform,
)
from model.models import get_model_arch
from utils.optimizers_shcedulers import CosineAnnealingLRWithWarmup, get_optimizer
from utils.tools import get_best_device, local_time


class MedFedAlignProClient:
    def __init__(self, args, dataset, client_id, logger):
        self.args = args
        self.dataset = dataset
        self.client_id = client_id
        self.logger = logger
        self.device = get_best_device(self.args.use_cuda)
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        self.classification_model = get_model_arch(model_name=self.args.medical_backbone)(
            dataset=MEDICAL_DATASET_NAME
        )
        self.optimizer = get_optimizer(
            self.classification_model,
            self.args.optimizer,
            self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = CosineAnnealingLRWithWarmup(
            optimizer=self.optimizer,
            total_epochs=max(self.args.num_epochs * self.args.round, 1),
        )
        self.augment_transform = get_medical_augment_transform(self.args.image_size)
        self.global_anchors = None
        self.global_anchor_mask = None

    def load_model_weights(self, model_weights):
        self.classification_model.load_state_dict(model_weights)

    def set_global_anchors(self, anchors, anchor_mask):
        self.global_anchors = anchors.clone().to(self.device)
        self.global_anchor_mask = anchor_mask.clone().to(self.device)

    def get_model_weights(self):
        return deepcopy(self.classification_model.state_dict())

    def move_to_device(self):
        self.device = get_best_device(self.args.use_cuda)
        self.classification_model.to(self.device)

    def apply_augmentation(self, batch: torch.Tensor) -> torch.Tensor:
        augmented = []
        for image in batch:
            aug_image = self.augment_transform(image.cpu())
            aug_image = aug_image + torch.randn_like(aug_image) * self.args.noise_std
            augmented.append(aug_image)
        return torch.stack(augmented, dim=0).to(self.device)

    def binary_js_divergence(self, logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
        probs_a = torch.sigmoid(logits_a).clamp(1e-6, 1 - 1e-6)
        probs_b = torch.sigmoid(logits_b).clamp(1e-6, 1 - 1e-6)
        dist_a = torch.stack([1 - probs_a, probs_a], dim=1)
        dist_b = torch.stack([1 - probs_b, probs_b], dim=1)
        mean_dist = 0.5 * (dist_a + dist_b)
        kl_a = (dist_a * (dist_a.log() - mean_dist.log())).sum(dim=1)
        kl_b = (dist_b * (dist_b.log() - mean_dist.log())).sum(dim=1)
        return 0.5 * (kl_a + kl_b).mean()

    def prototype_nce_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.global_anchors is None or self.global_anchor_mask is None:
            return torch.tensor(0.0, device=self.device)
        if int(self.global_anchor_mask.sum().item()) < 2:
            return torch.tensor(0.0, device=self.device)
        valid_sample_mask = self.global_anchor_mask[labels]
        if valid_sample_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        features = F.normalize(features[valid_sample_mask], dim=1)
        labels = labels[valid_sample_mask]
        anchors = F.normalize(self.global_anchors, dim=1)
        logits = torch.matmul(features, anchors.T) / self.args.proto_temperature
        invalid_mask = ~self.global_anchor_mask.unsqueeze(0)
        logits = logits.masked_fill(invalid_mask, -1e4)
        return F.cross_entropy(logits, labels)

    def anchor_pull_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.global_anchors is None or self.global_anchor_mask is None:
            return torch.tensor(0.0, device=self.device)
        pull_losses = []
        for class_id in range(MEDICAL_NUM_CLASSES):
            class_mask = labels == class_id
            if class_mask.sum() == 0 or not bool(self.global_anchor_mask[class_id]):
                continue
            batch_proto = features[class_mask].mean(dim=0)
            pull_losses.append(F.mse_loss(batch_proto, self.global_anchors[class_id]))
        if not pull_losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(pull_losses).mean()

    @torch.no_grad()
    def compute_local_prototypes(self):
        self.classification_model.eval()
        feature_dim = self.classification_model.feature_dim
        proto_sum = torch.zeros(MEDICAL_NUM_CLASSES, feature_dim, device=self.device)
        counts = torch.zeros(MEDICAL_NUM_CLASSES, device=self.device)
        confidence_sum = torch.zeros(MEDICAL_NUM_CLASSES, device=self.device)
        for data, target, _ in self.train_loader:
            data = data.to(self.device)
            target = target.to(self.device).long()
            features = self.classification_model.base(data)
            logits = self.classification_model.classifier(features).squeeze(-1)
            probs = torch.sigmoid(logits)
            confidence = torch.where(target == 1, probs, 1 - probs)
            for class_id in range(MEDICAL_NUM_CLASSES):
                class_mask = target == class_id
                if class_mask.sum() == 0:
                    continue
                proto_sum[class_id] += features[class_mask].sum(dim=0)
                counts[class_id] += class_mask.sum()
                confidence_sum[class_id] += confidence[class_mask].sum()

        prototypes = torch.zeros_like(proto_sum)
        reliabilities = torch.zeros_like(counts)
        valid_mask = counts >= float(self.args.min_class_count)
        prototypes[valid_mask] = proto_sum[valid_mask] / counts[valid_mask].unsqueeze(1).clamp_min(1.0)
        mean_conf = torch.zeros_like(confidence_sum)
        mean_conf[valid_mask] = confidence_sum[valid_mask] / counts[valid_mask].clamp_min(1.0)
        reliabilities[valid_mask] = torch.sqrt(counts[valid_mask]) * mean_conf[valid_mask]
        return {
            "prototypes": prototypes.detach().cpu(),
            "counts": counts.detach().cpu(),
            "reliabilities": reliabilities.detach().cpu(),
        }

    def train(self):
        self.move_to_device()
        self.classification_model.train()
        criterion = torch.nn.BCEWithLogitsLoss()
        total_loss = 0.0
        steps = 0
        for _ in range(self.args.num_epochs):
            for data, target, _ in self.train_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                target_long = target.long()
                augmented = self.apply_augmentation(data)
                self.optimizer.zero_grad()

                features = self.classification_model.base(data)
                logits = self.classification_model.classifier(features).squeeze(-1)

                aug_features = self.classification_model.base(augmented)
                aug_logits = self.classification_model.classifier(aug_features).squeeze(-1)

                loss_cls = criterion(logits, target)
                loss_js = self.binary_js_divergence(logits, aug_logits)
                loss_feat = F.mse_loss(features, aug_features)
                loss_proto = self.prototype_nce_loss(features, target_long)
                loss_anchor = self.anchor_pull_loss(features, target_long)
                loss = (
                    loss_cls
                    + self.args.lambda_pred * loss_js
                    + self.args.lambda_feat * loss_feat
                    + self.args.lambda_proto * loss_proto
                    + self.args.lambda_anchor * loss_anchor
                )
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                steps += 1
            self.scheduler.step()

        prototype_payload = self.compute_local_prototypes()
        self.classification_model.to(torch.device("cpu"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        average_loss = total_loss / max(steps, 1)
        self.logger.log(
            f"{local_time()}, Client {self.client_id}, Avg Loss: {average_loss:.4f}"
        )
        return {
            "weights": self.get_model_weights(),
            "num_samples": len(self.dataset),
            "avg_loss": average_loss,
            **prototype_payload,
        }
