from argparse import ArgumentParser, Namespace
from copy import deepcopy
import os
from pathlib import Path
import pickle
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
OUT_DIR = os.path.join(PROJECT_DIR, "out")
sys.path.append(PROJECT_DIR.as_posix())

from algorithm.client.medfedalignpro import MedFedAlignProClient
from data.medical_dataset import (
    DEFAULT_MEDICAL_DOMAINS,
    MEDICAL_DATASET_NAME,
    MEDICAL_NUM_CLASSES,
    NEGATIVE_LABEL_NAME,
    POSITIVE_LABEL_NAME,
    build_medical_federated_splits,
    load_medical_domain_bundle,
)
from model.models import get_model_arch
from utils.tools import Logger, fix_random_seed, get_best_device, local_time, str2bool


def get_medfedalignpro_argparser():
    parser = ArgumentParser(description="MedFedAlignPro arguments.")
    parser.add_argument("-d", "--dataset", type=str, default=MEDICAL_DATASET_NAME)
    parser.add_argument("--use-cuda", type=str2bool, default=True)
    parser.add_argument("--save_log", type=str2bool, default=True)
    parser.add_argument("--output_dir", type=str, default="medical_output")
    parser.add_argument("--medical_backbone", type=str, default="res18", choices=["res18", "res34", "res50", "mobile3s", "mobile3l"])
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--round", type=int, default=5)
    parser.add_argument("--test_gap", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_clients_per_domain", type=int, default=2)
    parser.add_argument("--lambda_proto", type=float, default=0.5)
    parser.add_argument("--lambda_anchor", type=float, default=0.25)
    parser.add_argument("--lambda_pred", type=float, default=0.1)
    parser.add_argument("--lambda_feat", type=float, default=0.1)
    parser.add_argument("--proto_momentum", type=float, default=0.8)
    parser.add_argument("--proto_temperature", type=float, default=0.1)
    parser.add_argument("--min_class_count", type=int, default=4)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--domains", type=str, default=",".join(DEFAULT_MEDICAL_DOMAINS))
    parser.add_argument("--heldout_domain", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--plot_tsne", action="store_true")
    parser.add_argument("--binary_task", type=str, default="referable_dr")
    return parser


class MedFedAlignProServer:
    def __init__(self, algo="MedFedAlignPro", args: Namespace = None):
        self.args = get_medfedalignpro_argparser().parse_args() if args is None else args
        self.algo = algo
        self.args.dataset = MEDICAL_DATASET_NAME
        if self.args.output_dir == "medical_output":
            self.args.output_dir = local_time()
        self.args.domains = [
            item.strip() for item in self.args.domains.split(",") if item.strip()
        ]
        fix_random_seed(self.args.seed)
        self.device = get_best_device(self.args.use_cuda)
        self.stores, self.samples_by_domain, self.unavailable_domains = load_medical_domain_bundle(
            domains=self.args.domains,
            cache_dir=self.args.cache_dir or None,
        )
        self.available_domains = sorted(self.samples_by_domain.keys())
        if len(self.available_domains) < 2:
            raise RuntimeError(
                "MedFedAlignPro requires at least two available domains. "
                f"Available: {self.available_domains}, unavailable: {self.unavailable_domains}"
            )
        self.summary_records = []

    def initialize_logger(self, heldout_domain: str):
        self.path2output_dir = os.path.join(
            OUT_DIR, self.algo, self.args.dataset, self.args.output_dir, heldout_domain
        )
        os.makedirs(self.path2output_dir, exist_ok=True)
        stdout = Console(log_path=False, log_time=False)
        logfile_path = os.path.join(self.path2output_dir, "log.html")
        self.logger = Logger(
            stdout=stdout,
            enable_log=self.args.save_log,
            logfile_path=logfile_path,
        )
        self.logger.log("=" * 20, self.algo, heldout_domain, "=" * 20)
        self.logger.log("Experiment Arguments:", dict(self.args._get_kwargs()))
        if self.unavailable_domains:
            self.logger.log("Unavailable domains:", self.unavailable_domains)

    def initialize_model(self):
        self.classification_model = get_model_arch(model_name=self.args.medical_backbone)(
            dataset=MEDICAL_DATASET_NAME
        )
        self.classification_model.to(self.device)
        feature_dim = self.classification_model.feature_dim
        self.global_anchors = torch.zeros(MEDICAL_NUM_CLASSES, feature_dim, device=self.device)
        self.global_anchor_mask = torch.zeros(MEDICAL_NUM_CLASSES, dtype=torch.bool, device=self.device)

    def initialize_run(self, heldout_domain: str):
        self.initialize_logger(heldout_domain)
        self.initialize_model()
        split_bundle = build_medical_federated_splits(
            stores=self.stores,
            samples_by_domain=self.samples_by_domain,
            heldout_domain=heldout_domain,
            num_clients_per_domain=self.args.num_clients_per_domain,
            val_ratio=self.args.val_ratio,
            seed=self.args.seed,
            image_size=self.args.image_size,
        )
        self.client_datasets = split_bundle["client_datasets"]
        self.validation_set = split_bundle["validation_dataset"]
        self.test_set = split_bundle["test_dataset"]
        self.client_domain_summary = split_bundle["client_domain_summary"]
        self.client_list = [
            MedFedAlignProClient(self.args, dataset, client_id, self.logger)
            for client_id, dataset in enumerate(self.client_datasets)
        ]
        self.round_history = []
        self.best_macro_f1 = -1.0

    def aggregate_model(self, client_results):
        total_samples = sum(result["num_samples"] for result in client_results)
        aggregated = {}
        for key in client_results[0]["weights"].keys():
            aggregated[key] = sum(
                result["weights"][key] * (result["num_samples"] / max(total_samples, 1))
                for result in client_results
            )
        return aggregated

    def aggregate_prototypes(self, client_results):
        feature_dim = self.classification_model.feature_dim
        numerator = torch.zeros(MEDICAL_NUM_CLASSES, feature_dim, device=self.device)
        denominator = torch.zeros(MEDICAL_NUM_CLASSES, device=self.device)
        for result in client_results:
            prototypes = result["prototypes"].to(self.device)
            counts = result["counts"].to(self.device)
            reliabilities = result["reliabilities"].to(self.device)
            weights = counts * reliabilities
            numerator += prototypes * weights.unsqueeze(1)
            denominator += weights

        valid_mask = denominator > 0
        for class_id in range(MEDICAL_NUM_CLASSES):
            if not bool(valid_mask[class_id]):
                continue
            new_anchor = numerator[class_id] / denominator[class_id].clamp_min(1e-6)
            if bool(self.global_anchor_mask[class_id]):
                self.global_anchors[class_id] = (
                    self.args.proto_momentum * self.global_anchors[class_id]
                    + (1 - self.args.proto_momentum) * new_anchor
                )
            else:
                self.global_anchors[class_id] = new_anchor
            self.global_anchor_mask[class_id] = True

    def evaluate(self, dataset, split_name: str):
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        self.classification_model.eval()
        self.classification_model.to(self.device)
        logits_list = []
        labels_list = []
        domain_names = []
        feature_list = []
        with torch.no_grad():
            for data, target, domains in dataloader:
                data = data.to(self.device)
                target = target.to(self.device)
                features = self.classification_model.base(data)
                logits = self.classification_model.classifier(features).squeeze(-1)
                logits_list.append(logits.cpu())
                labels_list.append(target.cpu())
                feature_list.append(features.cpu())
                domain_names.extend(list(domains))
        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0).numpy().astype(int)
        features = torch.cat(feature_list, dim=0).numpy()
        probs = torch.sigmoid(logits).numpy()
        preds = (probs >= 0.5).astype(int)
        metrics = {
            "split": split_name,
            "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "referable_dr_f1": float(f1_score(labels, preds, pos_label=1, average="binary", zero_division=0)),
            "num_samples": int(len(labels)),
        }
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        self.plot_confusion_matrix(cm, split_name)
        if split_name == "test":
            self.plot_prototype_heatmap()
            if self.args.plot_tsne and len(labels) > 10:
                self.plot_tsne(features, labels, domain_names, split_name)
        return metrics

    def plot_confusion_matrix(self, cm, split_name: str):
        fig, ax = plt.subplots(figsize=(4, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[NEGATIVE_LABEL_NAME, POSITIVE_LABEL_NAME],
        )
        disp.plot(ax=ax, colorbar=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.path2output_dir, f"{split_name}_confusion_matrix.png"))
        plt.close(fig)

    def plot_prototype_heatmap(self):
        anchors = self.global_anchors.detach().cpu().numpy()
        if not self.global_anchor_mask.any():
            return
        norm = np.linalg.norm(anchors, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        anchors = anchors / norm
        cosine = anchors @ anchors.T
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cosine, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([NEGATIVE_LABEL_NAME, POSITIVE_LABEL_NAME])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([NEGATIVE_LABEL_NAME, POSITIVE_LABEL_NAME])
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(self.path2output_dir, "prototype_heatmap.png"))
        plt.close(fig)

    def plot_round_curves(self):
        if not self.round_history:
            return
        history_df = pd.DataFrame(self.round_history)
        history_df.to_csv(os.path.join(self.path2output_dir, "round_metrics.csv"), index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history_df["round"], history_df["train_loss"], label="train_loss")
        ax.plot(history_df["round"], history_df["val_macro_f1"], label="val_macro_f1")
        ax.plot(history_df["round"], history_df["test_macro_f1"], label="test_macro_f1")
        ax.set_xlabel("Round")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.path2output_dir, "round_curves.png"))
        plt.close(fig)

    def plot_tsne(self, features, labels, domains, split_name: str):
        if len(features) < 10:
            return
        tsne = TSNE(n_components=2, random_state=self.args.seed)
        embedding = tsne.fit_transform(features)
        unique_domains = sorted(set(domains))
        markers = ["o", "s", "^", "x", "D"]
        fig, ax = plt.subplots(figsize=(7, 6))
        for domain_index, domain in enumerate(unique_domains):
            domain_mask = np.array([item == domain for item in domains])
            ax.scatter(
                embedding[domain_mask, 0],
                embedding[domain_mask, 1],
                c=np.array(labels)[domain_mask],
                cmap="coolwarm",
                marker=markers[domain_index % len(markers)],
                alpha=0.6,
                label=domain,
            )
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.path2output_dir, f"{split_name}_tsne.png"))
        plt.close(fig)

    def save_checkpoint(self, round_id: int):
        checkpoint = {
            "model": self.classification_model.state_dict(),
            "round": round_id,
            "anchors": self.global_anchors.detach().cpu(),
            "anchor_mask": self.global_anchor_mask.detach().cpu(),
        }
        torch.save(checkpoint, os.path.join(self.path2output_dir, "checkpoint.pth"))

    def process_single_domain(self, heldout_domain: str):
        self.initialize_run(heldout_domain)
        global_weights = deepcopy(self.classification_model.state_dict())
        for round_id in range(self.args.round):
            self.logger.log("=" * 20, f"Round {round_id}", "=" * 20)
            client_results = []
            for client in self.client_list:
                client.load_model_weights(deepcopy(global_weights))
                client.set_global_anchors(self.global_anchors, self.global_anchor_mask)
                client_results.append(client.train())
            global_weights = self.aggregate_model(client_results)
            self.classification_model.load_state_dict(global_weights)
            self.aggregate_prototypes(client_results)
            if (round_id + 1) % self.args.test_gap == 0:
                val_metrics = self.evaluate(self.validation_set, "validation")
                test_metrics = self.evaluate(self.test_set, "test")
                train_loss = float(np.mean([item["avg_loss"] for item in client_results]))
                record = {
                    "round": round_id,
                    "train_loss": train_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "test_referable_dr_f1": test_metrics["referable_dr_f1"],
                }
                self.round_history.append(record)
                self.logger.log(
                    f"{local_time()}, Validation Macro-F1: {val_metrics['macro_f1']:.4f}, "
                    f"Test Macro-F1: {test_metrics['macro_f1']:.4f}"
                )
                if test_metrics["macro_f1"] >= self.best_macro_f1:
                    self.best_macro_f1 = test_metrics["macro_f1"]
                    self.save_checkpoint(round_id)
                    with open(os.path.join(self.path2output_dir, "test_accuracy.pkl"), "wb") as handle:
                        pickle.dump(test_metrics["accuracy"] * 100.0, handle)

        self.plot_round_curves()
        if self.round_history:
            final_row = self.round_history[-1]
            summary = {
                "heldout_domain": heldout_domain,
                "accuracy": final_row["test_accuracy"],
                "macro_f1": final_row["test_macro_f1"],
                "referable_dr_f1": final_row["test_referable_dr_f1"],
                "num_source_domains": len(self.available_domains) - 1,
            }
            pd.DataFrame([summary]).to_csv(
                os.path.join(self.path2output_dir, "summary.csv"), index=False
            )
            self.summary_records.append(summary)
            return summary
        return None

    def run_all_domains(self):
        heldout_domains = [self.args.heldout_domain] if self.args.heldout_domain else self.available_domains
        for heldout_domain in heldout_domains:
            if heldout_domain not in self.available_domains:
                warnings.warn(f"Skipping unavailable held-out domain `{heldout_domain}`.")
                continue
            self.process_single_domain(heldout_domain)
        if not self.summary_records:
            return
        summary_df = pd.DataFrame(self.summary_records)
        summary_df["mean_accuracy"] = summary_df["accuracy"].mean()
        summary_df["mean_macro_f1"] = summary_df["macro_f1"].mean()
        summary_df["worst_macro_f1"] = summary_df["macro_f1"].min()
        summary_dir = os.path.join(OUT_DIR, self.algo, self.args.dataset, self.args.output_dir)
        os.makedirs(summary_dir, exist_ok=True)
        summary_df.to_csv(os.path.join(summary_dir, "overall_summary.csv"), index=False)


if __name__ == "__main__":
    server = MedFedAlignProServer()
    server.run_all_domains()
