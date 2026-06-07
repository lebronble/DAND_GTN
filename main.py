#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import math
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tqdm import tqdm

from model.loss_fusion_modules import AdaptiveLossFusion

try:
    from tensorboardX import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = BASE_DIR
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "config", "train_DAND_GTN_v1.yaml")
for path in (BASE_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(REPO_ROOT)


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def get_parser():
    parser = argparse.ArgumentParser(description="PG-DNDT + CSTA training")
    parser.add_argument("--work-dir", default=os.path.join(BASE_DIR, "work_dir"))
    parser.add_argument("-model_saved_name", "--model-saved-name", dest="model_saved_name", default=os.path.join(BASE_DIR, "runs"))
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to the configuration file")

    parser.add_argument("--phase", default="train", help="must be train or test")
    parser.add_argument("--save-score", type=str2bool, default=False, help="store classification score")

    parser.add_argument("--seed", type=int, default=1, help="random seed for pytorch")
    parser.add_argument("--log-interval", type=int, default=100, help="print interval")
    parser.add_argument("--save-interval", type=int, default=2, help="model save interval")
    parser.add_argument("--eval-interval", type=int, default=5, help="evaluation interval")
    parser.add_argument("--print-log", type=str2bool, default=True, help="print logging or not")
    parser.add_argument("--show-topk", type=int, default=[1, 2], nargs="+", help="top-k accuracy")

    parser.add_argument("--feeder", default="feeders.feeder_same_combine.Feeder", help="data loader")
    parser.add_argument("--num-worker", type=int, default=8, help="number of workers")
    parser.add_argument("--train-feeder-args", default=dict(), help="training data loader args")
    parser.add_argument("--test-feeder-args", default=dict(), help="test data loader args")

    parser.add_argument("--train_ratio", default=0.9)
    parser.add_argument("--val_ratio", default=0.0)
    parser.add_argument("--test_ratio", default=0.1)

    parser.add_argument("--model", default=None, help="model class path")
    parser.add_argument("--model-args", type=dict, default=dict(), help="model args")
    parser.add_argument("--weights", default=None, help="weights for initialization")
    parser.add_argument("--ignore-weights", type=str, default=[], nargs="+", help="ignored weights")

    parser.add_argument("--base-lr", type=float, default=0.01, help="initial learning rate")
    parser.add_argument("--step", type=int, default=[20, 40, 60, 80], nargs="+", help="lr milestones")
    parser.add_argument("--device", type=int, default=0, nargs="+", help="GPU ids")
    parser.add_argument("--optimizer", default="SGD", help="optimizer type")
    parser.add_argument("--nesterov", type=str2bool, default=False, help="use nesterov")
    parser.add_argument("--batch-size", type=int, default=2, help="training batch size")
    parser.add_argument("--test-batch-size", type=int, default=2, help="test batch size")
    parser.add_argument("--start-epoch", type=int, default=0, help="start epoch")
    parser.add_argument("--num-epoch", type=int, default=80, help="number of epochs")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="weight decay")
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--warm_up_epoch", type=int, default=0)

    parser.add_argument("--aff-loss-type", default="MSE", choices=["MSE", "SmoothL1"], help="affective regression loss")
    parser.add_argument("--aff-loss-ramp-epochs", type=int, default=10, help="epochs used to ramp in affective loss")
    parser.add_argument("--shuffle-affective-target", type=str2bool, default=False, help="shuffle affective targets during training")
    parser.add_argument("--loss-min-cls-weight", type=float, default=0.6, help="lower bound for classification loss weight")
    parser.add_argument("--loss-min-aff-weight", type=float, default=0.05, help="lower bound for affective loss weight")
    parser.add_argument("--loss-entropy-weight", type=float, default=0.0, help="regularize dynamic loss weights")
    parser.add_argument("--export-visuals", type=str2bool, default=True, help="save visualization figures after training")
    parser.add_argument("--export-confusion", type=str2bool, default=True, help="save confusion matrix artifacts after training")
    parser.add_argument("--export-analysis-data", type=str2bool, default=True, help="save raw DND/CSTA data after training")
    parser.add_argument("--visual-sample-index", type=int, default=0, help="sample index used for visualization export")
    parser.add_argument("--max-train-batches", type=int, default=0, help="debug only: limit train batches when > 0")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="debug only: limit eval batches when > 0")
    return parser


class Processor:
    def __init__(self, arg):
        self.arg = arg
        os.makedirs(arg.work_dir, exist_ok=True)
        os.makedirs(arg.model_saved_name, exist_ok=True)

        if arg.phase == "train":
            self.save_arg()
            if not arg.train_feeder_args.get("debug", False):
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, "train"), "train")
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, "val"), "val")
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, "debug"), "debug")
        else:
            self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, "test"), "test")

        self.global_step = 0
        self.best_acc = 0.0
        self.best_metrics = None
        self.last_eval_metrics = dict()
        self.train_acc_list = []
        self.eval_acc_list = []
        self.export_visuals = bool(self.arg.export_visuals)
        self.export_confusion = bool(self.arg.export_confusion)
        self.export_analysis_data = bool(self.arg.export_analysis_data)
        self.visual_sample_index = max(0, int(self.arg.visual_sample_index))
        self.best_ckpt_path = os.path.join(self.arg.model_saved_name, "best_PG_DNDT_CSTA.pt")
        self.visual_dir = os.path.join(self.arg.model_saved_name, "visuals")
        self.confusion_dir = os.path.join(self.arg.model_saved_name, "confusion")
        self.analysis_dir = os.path.join(self.arg.model_saved_name, "analysis_data")
        self.best_state_dict = None

        self.loss_fusion = AdaptiveLossFusion(
            aff_loss_type=self.arg.aff_loss_type,
            aff_loss_ramp_epochs=self.arg.aff_loss_ramp_epochs,
            loss_min_cls_weight=self.arg.loss_min_cls_weight,
            loss_min_aff_weight=self.arg.loss_min_aff_weight,
            loss_entropy_weight=self.arg.loss_entropy_weight,
        )

        self.load_data()
        self.load_model()
        self.load_optimizer()

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == "train":
            self.data_loader["train"] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed,
            )
        self.data_loader["test"] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed,
        )

    def load_model(self):
        if torch.cuda.is_available():
            output_device = self.arg.device[0] if isinstance(self.arg.device, list) else self.arg.device
            self.device = torch.device("cuda:{}".format(output_device))
            self.output_device = output_device
        else:
            self.device = torch.device("cpu")
            self.output_device = None

        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        if self.arg.config:
            shutil.copy2(self.arg.config, self.arg.work_dir)
        shutil.copy2(__file__, self.arg.work_dir)

        self.model = Model(**self.arg.model_args).to(self.device)

        if self.arg.weights:
            self.print_log("Load weights from {}.".format(self.arg.weights))
            if ".pkl" in self.arg.weights:
                with open(self.arg.weights, "rb") as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights, map_location="cpu")

            weights = OrderedDict([[k.split("module.")[-1], v] for k, v in weights.items()])
            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log("Successfully remove weights: {}.".format(key))
                        else:
                            self.print_log("Can not remove weights: {}.".format(key))

            try:
                self.model.load_state_dict(weights)
            except RuntimeError:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                self.print_log("Can not find these weights:")
                for d in diff:
                    self.print_log("  " + d, print_time=False)
                state.update(weights)
                self.model.load_state_dict(state)

        if torch.cuda.is_available() and isinstance(self.arg.device, list) and len(self.arg.device) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=self.output_device)

    def load_optimizer(self):
        if self.arg.optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay,
            )
        elif self.arg.optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
            )
        elif self.arg.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
            )
        else:
            raise ValueError("Unsupported optimizer: {}".format(self.arg.optimizer))
        self.lr = self.arg.base_lr

    def save_arg(self):
        with open(os.path.join(self.arg.work_dir, "config.yaml"), "w") as f:
            yaml.dump(vars(self.arg), f)

    def adjust_learning_rate(self, epoch):
        warm_up_epoch = int(self.arg.warm_up_epoch)
        if warm_up_epoch > 0 and epoch < warm_up_epoch:
            lr = self.arg.base_lr * (epoch + 1) / warm_up_epoch
        else:
            lr = self.arg.base_lr * (0.1 ** np.sum(epoch >= np.array(self.arg.step)))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.lr = lr
        return lr

    @staticmethod
    def parse_model_output(output):
        return AdaptiveLossFusion.parse_model_output(output)

    def get_aff_loss_scale(self, epoch):
        target = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(target, "dla_head") and hasattr(target.dla_head, "get_aff_loss_scale"):
            return target.dla_head.get_aff_loss_scale(epoch)
        return self.loss_fusion.get_aff_loss_scale(epoch)

    def compute_losses(self, output, label, feature, epoch):
        target = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(target, "dla_head") and hasattr(target.dla_head, "compute_loss"):
            return target.dla_head.compute_loss(
                output,
                label,
                feature,
                epoch,
                training=self.model.training,
                shuffle_affective_target=self.arg.shuffle_affective_target,
            )
        return self.loss_fusion(
            output,
            label,
            feature,
            epoch,
            training=self.model.training,
            shuffle_affective_target=self.arg.shuffle_affective_target,
        )

    def load_checkpoint(self, path):
        if not os.path.isfile(path):
            return False
        state_dict = torch.load(path, map_location=self.device)
        if len(state_dict) > 0 and next(iter(state_dict.keys())).startswith("module."):
            state_dict = OrderedDict([[k.split("module.")[-1], v] for k, v in state_dict.items()])
        target = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        target.load_state_dict(state_dict)
        return True

    def _capture_model_state(self):
        target = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        return OrderedDict((k, v.detach().cpu().clone()) for k, v in target.state_dict().items())

    def _load_best_state_for_export(self):
        if self.best_state_dict is not None:
            target = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            target.load_state_dict(self.best_state_dict)
            self.print_log("Loaded in-memory best state for export.")
            return True
        if os.path.isfile(self.best_ckpt_path):
            self.load_checkpoint(self.best_ckpt_path)
            self.print_log("Loaded best checkpoint for export: {}".format(self.best_ckpt_path))
            return True
        self.print_log("Best checkpoint not found, exporting from current model state.")
        return False

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().float().numpy()
        return np.asarray(value)

    def _save_bar_figure(self, values, labels, title, path, color="#4C72B0"):
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        xs = np.arange(len(values))
        ax.bar(xs, values, color=color, width=0.65)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, max(1.0, float(values.max()) * 1.2 if values.size else 1.0))
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        for idx, val in enumerate(values):
            ax.text(idx, val + 0.02, "{:.2f}".format(float(val)), ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _save_weight_figure(self, stream_weights, loss_weights, path, title):
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
        stream_weights = np.asarray(stream_weights, dtype=np.float32).reshape(-1)
        loss_weights = np.asarray(loss_weights, dtype=np.float32).reshape(-1)
        panels = [
            (axes[0], stream_weights, ["Pose", "Motion"], "Stream weights", "#4C72B0"),
            (axes[1], loss_weights, ["Cls", "Aff"], "Loss weights", "#DD8452"),
        ]
        for ax, values, labels, panel_title, color in panels:
            xs = np.arange(len(values))
            ax.bar(xs, values, color=color, width=0.65)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels)
            ax.set_ylim(0.0, 1.0)
            ax.set_title(panel_title)
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            for idx, val in enumerate(values):
                ax.text(idx, val + 0.02, "{:.2f}".format(float(val)), ha="center", va="bottom", fontsize=8)
        fig.suptitle(title, y=1.02)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _save_relay_summary(self, branch_name, stage_aux_list, path):
        joint_count = int(self.arg.model_args.get("num_point", 16))
        joint_ids = np.arange(joint_count)
        fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex=True, sharey=True)
        axes = axes.flatten()
        stage_labels = ["Stage {}".format(i + 1) for i in range(len(stage_aux_list))]

        for idx, (ax, stage_aux) in enumerate(zip(axes, stage_aux_list)):
            relay_index = self._to_numpy(stage_aux["relay_index"]).astype(np.int64).reshape(-1)
            relay_importance = self._to_numpy(stage_aux["relay_importance"])
            if relay_importance.ndim > 1:
                relay_importance = relay_importance.mean(axis=0)
            counts = np.bincount(relay_index, minlength=joint_count).astype(np.float32)
            if counts.max() > 0:
                counts = counts / counts.max()
            if relay_importance.max() > 0:
                relay_importance = relay_importance / relay_importance.max()
            ax.bar(joint_ids, counts, color="#4C72B0", alpha=0.8, label="Relay freq")
            ax.plot(joint_ids, relay_importance, color="#DD8452", marker="o", linewidth=1.2, markersize=3, label="Importance")
            top_k = max(1, int(stage_aux["relay_index"].shape[-1]))
            top_nodes = np.argsort(counts)[-top_k:][::-1].tolist()
            ax.set_title("{} top {}".format(stage_labels[idx], top_nodes))
            ax.set_ylim(0.0, 1.05)
            ax.set_xticks(joint_ids)
            ax.set_xticklabels([str(i) for i in joint_ids], fontsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.2)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.suptitle("{} relay selection".format(branch_name), y=1.02)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _save_csta_stage(self, stage_idx, stage_aux, path):
        cross_xy = self._to_numpy(stage_aux["cross_attn_pose_to_motion"])
        cross_yx = self._to_numpy(stage_aux["cross_attn_motion_to_pose"])
        spatial_p = self._to_numpy(stage_aux["spatial_attn_pose"])
        spatial_m = self._to_numpy(stage_aux["spatial_attn_motion"])

        if cross_xy.ndim == 3:
            cross_xy = cross_xy.mean(axis=0)
        if cross_yx.ndim == 3:
            cross_yx = cross_yx.mean(axis=0)
        if spatial_p.ndim > 1:
            spatial_p = spatial_p.mean(axis=0)
        if spatial_m.ndim > 1:
            spatial_m = spatial_m.mean(axis=0)

        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
        heatmaps = [
            (axes[0, 0], cross_xy, "Pose -> Motion", "mako"),
            (axes[0, 1], cross_yx, "Motion -> Pose", "flare"),
        ]
        for ax, matrix, title, cmap in heatmaps:
            sns.heatmap(matrix, ax=ax, cmap=cmap, cbar=True, square=True)
            ax.set_title(title)
            ax.set_xlabel("Key frame")
            ax.set_ylabel("Query frame")

        joint_ids = np.arange(len(spatial_p))
        axes[1, 0].bar(joint_ids, spatial_p, color="#4C72B0", width=0.65)
        axes[1, 0].set_title("Pose spatial attention")
        axes[1, 0].set_xticks(joint_ids)
        axes[1, 0].set_xticklabels([str(i) for i in joint_ids], fontsize=8)
        axes[1, 0].set_ylim(0.0, max(1.0, float(spatial_p.max()) * 1.2 if spatial_p.size else 1.0))
        axes[1, 0].grid(axis="y", linestyle="--", alpha=0.25)

        axes[1, 1].bar(joint_ids, spatial_m, color="#55A868", width=0.65)
        axes[1, 1].set_title("Motion spatial attention")
        axes[1, 1].set_xticks(joint_ids)
        axes[1, 1].set_xticklabels([str(i) for i in joint_ids], fontsize=8)
        axes[1, 1].set_ylim(0.0, max(1.0, float(spatial_m.max()) * 1.2 if spatial_m.size else 1.0))
        axes[1, 1].grid(axis="y", linestyle="--", alpha=0.25)

        fig.suptitle("CSTA after stage {}".format(stage_idx), y=1.02)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _save_confusion_figure(self, confusion, class_names, path, title):
        confusion = np.asarray(confusion, dtype=np.float32)
        row_sum = confusion.sum(axis=1, keepdims=True)
        norm_confusion = np.divide(confusion, row_sum, out=np.zeros_like(confusion), where=row_sum > 0)
        annot = np.empty(confusion.shape, dtype=object)
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                annot[i, j] = "{}\n{:.1f}%".format(int(confusion[i, j]), float(norm_confusion[i, j] * 100.0))

        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        sns.heatmap(
            norm_confusion,
            ax=ax,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            annot=annot,
            fmt="",
            cbar=True,
            square=True,
            linewidths=0.5,
            linecolor="white",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _save_confusion_artifacts(self, prefix, y_true, y_pred, metric_values, class_names, title):
        os.makedirs(self.confusion_dir, exist_ok=True)
        confusion = np.asarray(metric_values["confusion"], dtype=np.int64)
        row_sum = confusion.sum(axis=1, keepdims=True).astype(np.float32)
        norm_confusion = np.divide(
            confusion.astype(np.float32),
            row_sum,
            out=np.zeros_like(confusion, dtype=np.float32),
            where=row_sum > 0,
        )
        np.savez_compressed(
            prefix + ".npz",
            y_true=np.asarray(y_true, dtype=np.int64),
            y_pred=np.asarray(y_pred, dtype=np.int64),
            confusion=confusion,
            confusion_norm=norm_confusion,
            class_names=np.asarray(class_names),
            accuracy=np.array(metric_values["accuracy"], dtype=np.float32),
            precision=np.array(metric_values["precision"], dtype=np.float32),
            recall=np.array(metric_values["recall"], dtype=np.float32),
            f1=np.array(metric_values["f1"], dtype=np.float32),
            gmean=np.array(metric_values["gmean"], dtype=np.float32),
            kappa=np.array(metric_values["kappa"], dtype=np.float32),
            epoch=np.array(metric_values.get("epoch", -1), dtype=np.int64),
        )
        np.savetxt(prefix + "_raw.csv", confusion, fmt="%d", delimiter=",")
        self._save_confusion_figure(confusion, class_names, prefix + ".png", title)

    def _relay_top_nodes(self, relay_index, joint_count):
        relay_index = np.asarray(relay_index, dtype=np.int64)
        flat = relay_index.reshape(-1)
        counts = np.bincount(flat, minlength=joint_count).astype(np.int64)
        top_k = relay_index.shape[-1] if relay_index.ndim > 0 else 1
        top_k = max(1, min(int(top_k), joint_count))
        top_nodes = np.argsort(counts)[-top_k:][::-1].astype(np.int64)
        unique_nodes = np.unique(flat).astype(np.int64)
        return counts, top_nodes, unique_nodes

    def export_stage_analysis(self):
        if not self.export_analysis_data:
            return

        os.makedirs(self.analysis_dir, exist_ok=True)
        self._load_best_state_for_export()

        loader = self.data_loader["test"]
        batch = next(iter(loader))
        data_m, data_p, label, feature, index = batch
        sample_idx = min(self.visual_sample_index, data_p.size(0) - 1)

        data_m = data_m[sample_idx : sample_idx + 1].float().to(self.device)
        data_p = data_p[sample_idx : sample_idx + 1].float().to(self.device)
        label = label[sample_idx : sample_idx + 1].long().to(self.device)
        if torch.is_tensor(index):
            sample_index = int(index.view(-1)[sample_idx].item())
        else:
            sample_index = int(np.asarray(index).reshape(-1)[sample_idx])

        prev_mode = self.model.training
        self.model.eval()
        with torch.no_grad():
            output = self.model(data_p, data_m, label=label, collect_visuals=True)

        logits, aff_pred, aux = self.parse_model_output(output)
        pred = int(torch.argmax(logits, dim=1).item())
        true = int(label.item())

        visuals = aux.get("visuals", {})
        pose_dndt = visuals.get("pose_dndt", [])
        motion_dndt = visuals.get("motion_dndt", [])
        csta_visuals = visuals.get("csta", {})
        joint_count = int(self.arg.model_args.get("num_point", 16))

        payload = {
            "sample_index": np.array(sample_index, dtype=np.int64),
            "sample_pos_in_batch": np.array(sample_idx, dtype=np.int64),
            "true_label": np.array(true, dtype=np.int64),
            "pred_label": np.array(pred, dtype=np.int64),
            "class_names": np.asarray(["Happy", "Sad", "Angry", "Neutral"]),
            "stream_weights": self._to_numpy(aux["stream_weights"][0]),
            "loss_weights": self._to_numpy(aux["loss_weights"][0]),
            "final_context": self._to_numpy(aux["final_context"][0]),
        }

        summary_lines = [
            "sample_index: {}".format(sample_index),
            "sample_pos_in_batch: {}".format(sample_idx),
            "true_label: {}".format(true),
            "pred_label: {}".format(pred),
            "stream_weights: {}".format(self._to_numpy(aux["stream_weights"][0]).tolist()),
            "loss_weights: {}".format(self._to_numpy(aux["loss_weights"][0]).tolist()),
        ]

        for branch_name, stage_list in (("pose", pose_dndt), ("motion", motion_dndt)):
            for stage_idx, stage_aux in enumerate(stage_list, start=1):
                relay_index = self._to_numpy(stage_aux["relay_index"]).astype(np.int64)
                relay_importance = self._to_numpy(stage_aux["relay_importance"]).astype(np.float32)
                group = self._to_numpy(stage_aux["group"]).astype(np.float32)
                counts = self._to_numpy(stage_aux["counts"]).astype(np.float32)
                relay_freq, top_nodes, unique_nodes = self._relay_top_nodes(relay_index, joint_count)

                prefix = "{}_stage_{}".format(branch_name, stage_idx)
                payload["{}_relay_index".format(prefix)] = relay_index
                payload["{}_relay_importance".format(prefix)] = relay_importance
                payload["{}_relay_group".format(prefix)] = group
                payload["{}_relay_counts".format(prefix)] = counts
                payload["{}_relay_frequency".format(prefix)] = relay_freq
                payload["{}_relay_top_nodes".format(prefix)] = top_nodes
                payload["{}_relay_unique_nodes".format(prefix)] = unique_nodes

                summary_lines.append(
                    "{} relay_index_shape: {}".format(prefix, list(relay_index.shape))
                )
                summary_lines.append(
                    "{} relay_top_nodes: {}".format(prefix, top_nodes.tolist())
                )
                summary_lines.append(
                    "{} relay_unique_nodes: {}".format(prefix, unique_nodes.tolist())
                )

        for stage_key in sorted(csta_visuals.keys(), key=lambda x: int(x)):
            stage_aux = csta_visuals[stage_key]
            cross_xy = self._to_numpy(stage_aux["cross_attn_pose_to_motion"])
            cross_yx = self._to_numpy(stage_aux["cross_attn_motion_to_pose"])
            spatial_p = self._to_numpy(stage_aux["spatial_attn_pose"])
            spatial_m = self._to_numpy(stage_aux["spatial_attn_motion"])

            if cross_xy.ndim == 3:
                cross_xy_mean = cross_xy.mean(axis=0)
            else:
                cross_xy_mean = cross_xy
            if cross_yx.ndim == 3:
                cross_yx_mean = cross_yx.mean(axis=0)
            else:
                cross_yx_mean = cross_yx
            if spatial_p.ndim > 1:
                spatial_p_mean = spatial_p.mean(axis=0)
            else:
                spatial_p_mean = spatial_p
            if spatial_m.ndim > 1:
                spatial_m_mean = spatial_m.mean(axis=0)
            else:
                spatial_m_mean = spatial_m

            prefix = "csta_stage_{}".format(stage_key)
            payload["{}_pose_to_motion".format(prefix)] = cross_xy
            payload["{}_motion_to_pose".format(prefix)] = cross_yx
            payload["{}_pose_to_motion_mean".format(prefix)] = cross_xy_mean
            payload["{}_motion_to_pose_mean".format(prefix)] = cross_yx_mean
            payload["{}_pose_spatial".format(prefix)] = spatial_p
            payload["{}_motion_spatial".format(prefix)] = spatial_m
            payload["{}_pose_spatial_mean".format(prefix)] = spatial_p_mean
            payload["{}_motion_spatial_mean".format(prefix)] = spatial_m_mean
            payload["{}_attention_pair".format(prefix)] = np.stack([spatial_p_mean, spatial_m_mean], axis=0)

            summary_lines.append(
                "{} pose_to_motion_shape: {}".format(prefix, list(cross_xy.shape))
            )
            summary_lines.append(
                "{} motion_to_pose_shape: {}".format(prefix, list(cross_yx.shape))
            )
            summary_lines.append(
                "{} pose_spatial_shape: {}".format(prefix, list(spatial_p.shape))
            )
            summary_lines.append(
                "{} motion_spatial_shape: {}".format(prefix, list(spatial_m.shape))
            )

        np.savez_compressed(
            os.path.join(self.analysis_dir, "best_stage_analysis.npz"),
            **payload,
        )
        with open(os.path.join(self.analysis_dir, "best_stage_analysis_summary.txt"), "w") as f:
            f.write("\n".join(summary_lines))

        self.print_log("Stage analysis data saved to {}".format(self.analysis_dir))
        if prev_mode:
            self.model.train()

    def export_visualizations(self):
        if not self.export_visuals:
            return

        os.makedirs(self.visual_dir, exist_ok=True)
        self._load_best_state_for_export()

        loader = self.data_loader["test"]
        batch = next(iter(loader))
        data_m, data_p, label, feature, index = batch
        sample_idx = min(self.visual_sample_index, data_p.size(0) - 1)

        data_m = data_m[sample_idx : sample_idx + 1].float().to(self.device)
        data_p = data_p[sample_idx : sample_idx + 1].float().to(self.device)
        label = label[sample_idx : sample_idx + 1].long().to(self.device)
        feature = feature[sample_idx : sample_idx + 1].float().to(self.device)
        if torch.is_tensor(index):
            sample_index = int(index.view(-1)[sample_idx].item())
        else:
            sample_index = int(np.asarray(index).reshape(-1)[sample_idx])

        prev_mode = self.model.training
        self.model.eval()
        with torch.no_grad():
            output = self.model(data_p, data_m, label=label, collect_visuals=True)
            logits, loss, metrics = self.compute_losses(output, label, feature, self.best_metrics["epoch"] - 1 if self.best_metrics else 0)

        logits, aff_pred, aux = self.parse_model_output(output)
        pred = int(torch.argmax(logits, dim=1).item())
        true = int(label.item())
        stream_weights = self._to_numpy(aux["stream_weights"][0])
        loss_weights = np.asarray(
            [
                float(metrics["loss_weight_cls"].item()),
                float(metrics["loss_weight_aff"].item()),
            ],
            dtype=np.float32,
        )

        self._save_weight_figure(
            stream_weights,
            loss_weights,
            os.path.join(self.visual_dir, "sample_weights.png"),
            "Sample {} weights (true {}, pred {})".format(sample_index, true, pred),
        )

        visuals = aux.get("visuals", {})
        pose_dndt = visuals.get("pose_dndt", [])
        motion_dndt = visuals.get("motion_dndt", [])
        csta_visuals = visuals.get("csta", {})

        if pose_dndt:
            self._save_relay_summary("Pose branch", pose_dndt, os.path.join(self.visual_dir, "pose_relay_summary.png"))
        if motion_dndt:
            self._save_relay_summary("Motion branch", motion_dndt, os.path.join(self.visual_dir, "motion_relay_summary.png"))
        for stage_key in sorted(csta_visuals.keys(), key=lambda x: int(x)):
            self._save_csta_stage(
                int(stage_key),
                csta_visuals[stage_key],
                os.path.join(self.visual_dir, "csta_stage_{}.png".format(stage_key)),
            )

        summary_path = os.path.join(self.visual_dir, "visualization_summary.txt")
        class_names = ["Happy", "Neutral", "Angry", "Sad"]
        if len(class_names) <= max(true, pred):
            class_names = ["Class{}".format(i) for i in range(max(true, pred) + 1)]
        with open(summary_path, "w") as f:
            f.write("sample_index: {}\n".format(sample_index))
            f.write("true_label: {} ({})\n".format(true, class_names[true]))
            f.write("pred_label: {} ({})\n".format(pred, class_names[pred]))
            f.write("stream_weights: pose {:.4f}, motion {:.4f}\n".format(float(stream_weights[0]), float(stream_weights[1])))
            f.write("loss_weights: cls {:.4f}, aff {:.4f}\n".format(float(loss_weights[0]), float(loss_weights[1])))
            if pose_dndt:
                for stage_idx, stage_aux in enumerate(pose_dndt, start=1):
                    relay_index = self._to_numpy(stage_aux["relay_index"]).astype(np.int64).reshape(-1)
                    counts = np.bincount(relay_index, minlength=int(self.arg.model_args.get("num_point", 16))).astype(np.float32)
                    if counts.max() > 0:
                        counts = counts / counts.max()
                    top_nodes = np.argsort(counts)[-max(1, int(stage_aux["relay_index"].shape[-1])):][::-1].tolist()
                    f.write("pose_stage_{} relay_top_nodes: {}\n".format(stage_idx, top_nodes))
            if motion_dndt:
                for stage_idx, stage_aux in enumerate(motion_dndt, start=1):
                    relay_index = self._to_numpy(stage_aux["relay_index"]).astype(np.int64).reshape(-1)
                    counts = np.bincount(relay_index, minlength=int(self.arg.model_args.get("num_point", 16))).astype(np.float32)
                    if counts.max() > 0:
                        counts = counts / counts.max()
                    top_nodes = np.argsort(counts)[-max(1, int(stage_aux["relay_index"].shape[-1])):][::-1].tolist()
                    f.write("motion_stage_{} relay_top_nodes: {}\n".format(stage_idx, top_nodes))

        self.print_log("Visualization files saved to {}".format(self.visual_dir))
        if prev_mode:
            self.model.train()

    def export_confusion_matrix(self):
        if not self.export_confusion:
            return

        self._load_best_state_for_export()
        prev_mode = self.model.training
        self.model.eval()

        num_class = int(self.arg.model_args.get("num_class", 4))
        all_true = []
        all_pred = []
        process = tqdm(self.data_loader["test"])
        for data_m, data_p, label, feature, index in process:
            with torch.no_grad():
                data_m = data_m.float().to(self.device)
                data_p = data_p.float().to(self.device)
                label = label.long().to(self.device)

                output = self.model(data_p, data_m, label=label)
                logits, aff_pred, aux = self.parse_model_output(output)
                pred_label = torch.argmax(logits, dim=1)

                all_true.extend(label.detach().cpu().numpy().tolist())
                all_pred.extend(pred_label.detach().cpu().numpy().tolist())

        metric_values = self.classification_metrics(all_true, all_pred, num_class)
        class_names = ["Happy", "Neutral", "Angry", "Sad"]
        if len(class_names) != num_class:
            class_names = ["Class{}".format(i) for i in range(num_class)]

        prefix = os.path.join(self.confusion_dir, "confusion_best")
        title = "Best model confusion matrix"
        self._save_confusion_artifacts(prefix, all_true, all_pred, metric_values, class_names, title)
        self.print_log("Confusion matrix files saved to {}".format(self.confusion_dir))

        if prev_mode:
            self.model.train()

    def print_log(self, text, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            text = "[ " + localtime + " ] " + text
        print(text)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, "log.txt"), "a") as f:
                print(text, file=f)

    @staticmethod
    def class_accuracy_text(true_num, total_num):
        names = ["Happy", "Neutral", "Angry", "Sad"]
        if len(total_num) != len(names):
            names = ["Class{}".format(i) for i in range(len(total_num))]
        values = []
        for name, true_count, total_count in zip(names, true_num, total_num):
            if total_count == 0:
                values.append("{}:nan".format(name))
            else:
                values.append("{}:{:.4f}".format(name, true_count * 1.0 / total_count))
        return ",".join(values)

    @staticmethod
    def classification_metrics(y_true, y_pred, num_class):
        confusion = np.zeros((num_class, num_class), dtype=np.int64)
        for true, pred in zip(y_true, y_pred):
            true = int(true)
            pred = int(pred)
            if 0 <= true < num_class and 0 <= pred < num_class:
                confusion[true, pred] += 1

        total = int(confusion.sum())
        if total == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "gmean": 0.0,
                "kappa": 0.0,
                "confusion": confusion,
            }

        tp = np.diag(confusion).astype(np.float64)
        support = confusion.sum(axis=1).astype(np.float64)
        pred_count = confusion.sum(axis=0).astype(np.float64)
        valid = support > 0

        precision_per_class = np.zeros(num_class, dtype=np.float64)
        recall_per_class = np.zeros(num_class, dtype=np.float64)
        np.divide(tp, pred_count, out=precision_per_class, where=pred_count > 0)
        np.divide(tp, support, out=recall_per_class, where=support > 0)

        denom = precision_per_class + recall_per_class
        f1_per_class = np.zeros(num_class, dtype=np.float64)
        np.divide(
            2.0 * precision_per_class * recall_per_class,
            denom,
            out=f1_per_class,
            where=denom > 0,
        )

        precision = float(np.mean(precision_per_class[valid])) if np.any(valid) else 0.0
        recall = float(np.mean(recall_per_class[valid])) if np.any(valid) else 0.0
        f1 = float(np.mean(f1_per_class[valid])) if np.any(valid) else 0.0

        recalls = recall_per_class[valid]
        if recalls.size == 0 or np.any(recalls <= 0):
            gmean = 0.0
        else:
            gmean = float(np.exp(np.mean(np.log(recalls))))

        accuracy = float(tp.sum() / total)
        true_marginal = confusion.sum(axis=1).astype(np.float64)
        pred_marginal = confusion.sum(axis=0).astype(np.float64)
        pe = float(np.dot(true_marginal, pred_marginal) / (total * total))
        kappa = float((accuracy - pe) / (1.0 - pe)) if abs(1.0 - pe) > 1e-12 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "gmean": gmean,
            "kappa": kappa,
            "confusion": confusion,
        }

    @staticmethod
    def format_seconds(seconds):
        seconds = int(round(float(seconds)))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

    @staticmethod
    def format_large_number(value, suffix):
        if value is None:
            return "N/A"
        value = float(value)
        if abs(value) >= 1e9:
            return "{:.3f}G{}".format(value / 1e9, suffix)
        if abs(value) >= 1e6:
            return "{:.3f}M{}".format(value / 1e6, suffix)
        if abs(value) >= 1e3:
            return "{:.3f}K{}".format(value / 1e3, suffix)
        return "{:.0f}{}".format(value, suffix)

    def model_parameter_count(self):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def estimate_flops(self):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        try:
            sample = next(iter(self.data_loader["test"]))
        except StopIteration:
            return None

        data_m, data_p, label, feature, index = sample
        batch_size = min(1, data_p.size(0))
        data_m = data_m[:batch_size].float().to(self.device)
        data_p = data_p[:batch_size].float().to(self.device)

        flops = {"total": 0.0}
        handles = []

        def add_flops(value):
            flops["total"] += float(value)

        def conv_hook(module, inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if not torch.is_tensor(output):
                return
            kernel_ops = module.weight[0].numel()
            add_flops(output.numel() * kernel_ops)
            if module.bias is not None:
                add_flops(output.numel())

        def linear_hook(module, inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if not torch.is_tensor(output):
                return
            add_flops(output.numel() * module.in_features)
            if module.bias is not None:
                add_flops(output.numel())

        def bn_hook(module, inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if torch.is_tensor(output):
                add_flops(output.numel() * 2)

        def mha_hook(module, inputs, output):
            if len(inputs) < 3:
                return
            query, key, value = inputs[:3]
            if not (torch.is_tensor(query) and torch.is_tensor(key) and torch.is_tensor(value)):
                return
            if module.batch_first:
                batch = query.size(0)
                q_len = query.size(1)
                k_len = key.size(1)
            else:
                q_len = query.size(0)
                batch = query.size(1)
                k_len = key.size(0)
            embed_dim = module.embed_dim
            num_heads = module.num_heads
            head_dim = embed_dim // max(1, num_heads)
            # Q/K/V projections + attention scores + attention-value product + output projection.
            proj_flops = batch * (q_len + 2 * k_len) * embed_dim * embed_dim
            attn_flops = 2 * batch * num_heads * q_len * k_len * head_dim
            out_proj_flops = batch * q_len * embed_dim * embed_dim
            add_flops(proj_flops + attn_flops + out_proj_flops)

        for module in model.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                handles.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(linear_hook))
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                handles.append(module.register_forward_hook(bn_hook))
            elif isinstance(module, nn.MultiheadAttention):
                handles.append(module.register_forward_hook(mha_hook))

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                model(data_p, data_m)
        finally:
            for handle in handles:
                handle.remove()
            if was_training:
                model.train()

        return flops["total"] / batch_size

    def print_final_summary(self, runtime_seconds):
        total_params, trainable_params = self.model_parameter_count()
        try:
            flops = self.estimate_flops()
        except Exception as exc:
            flops = None
            self.print_log("FLOPs estimation failed: {}".format(exc))

        metrics = self.best_metrics or self.last_eval_metrics.get("test")
        self.print_log("=" * 72)
        self.print_log("Final Training Summary", print_time=False)
        if metrics is not None:
            self.print_log("Best Epoch: {}".format(metrics.get("epoch", "N/A")), print_time=False)
            self.print_log("Acc: {:.2f}%".format(metrics["accuracy"] * 100), print_time=False)
            self.print_log("Precision: {:.2f}%".format(metrics["precision"] * 100), print_time=False)
            self.print_log("Recall: {:.2f}%".format(metrics["recall"] * 100), print_time=False)
            self.print_log("F1: {:.4f}".format(metrics["f1"]), print_time=False)
            self.print_log("G-mean: {:.2f}%".format(metrics["gmean"] * 100), print_time=False)
            self.print_log("Kappa: {:.4f}".format(metrics["kappa"]), print_time=False)
            self.print_log(
                "Sub-class Acc: {}".format(metrics.get("class_accuracy_text", "N/A")),
                print_time=False,
            )
        else:
            self.print_log("Acc/F1/G-mean/Kappa: N/A (no eval result)", print_time=False)
        self.print_log(
            "Params: {} total / {} trainable".format(
                self.format_large_number(total_params, ""),
                self.format_large_number(trainable_params, ""),
            ),
            print_time=False,
        )
        self.print_log("FLOPs: {} per sample (approx.)".format(self.format_large_number(flops, "F")), print_time=False)
        self.print_log("Runtime: {}".format(self.format_seconds(runtime_seconds)), print_time=False)
        self.print_log("=" * 72, print_time=False)

    def train(self, epoch):
        self.model.train()
        self.print_log("Training epoch: {}".format(epoch + 1))
        self.adjust_learning_rate(epoch)
        self.train_writer.add_scalar("epoch", epoch, self.global_step)

        num_class = int(self.arg.model_args.get("num_class", 4))
        class_total = np.zeros(num_class, dtype=np.int64)
        class_true = np.zeros(num_class, dtype=np.int64)
        total_correct = 0
        total_count = 0
        loss_values = []

        process = tqdm(self.data_loader["train"])
        for batch_idx, (data_m, data_p, label, feature, index) in enumerate(process):
            self.global_step += 1
            data_m = data_m.float().to(self.device)
            data_p = data_p.float().to(self.device)
            label = label.long().to(self.device)
            feature = feature.float().to(self.device)

            output = self.model(data_p, data_m, label=label)
            logits, loss, metrics = self.compute_losses(output, label, feature, epoch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, predict_label = torch.max(logits.data, 1)
            total_correct += int((predict_label == label).sum().item())
            total_count += label.size(0)
            for pred, true in zip(predict_label.detach().cpu().numpy(), label.detach().cpu().numpy()):
                class_total[true] += 1
                class_true[true] += int(pred == true)

            loss_values.append(loss.item())
            self.train_writer.add_scalar("loss", loss.item(), self.global_step)
            self.train_writer.add_scalar("loss_cls", metrics["loss_cls"].item(), self.global_step)
            self.train_writer.add_scalar("loss_aff", metrics["loss_aff"].item(), self.global_step)
            self.train_writer.add_scalar("loss_weight_cls", metrics["loss_weight_cls"].item(), self.global_step)
            self.train_writer.add_scalar("loss_weight_aff", metrics["loss_weight_aff"].item(), self.global_step)
            self.train_writer.add_scalar("aff_scale", metrics["aff_scale"].item(), self.global_step)
            self.train_writer.add_scalar("lr", self.lr, self.global_step)

            if batch_idx % max(1, self.arg.log_interval) == 0:
                acc = 100.0 * total_correct / max(1, total_count)
                process.set_description("loss {:.4f} acc {:.2f}".format(loss.item(), acc))
            if self.arg.max_train_batches > 0 and batch_idx + 1 >= self.arg.max_train_batches:
                break

        train_acc = total_correct * 1.0 / max(1, total_count)
        self.train_acc_list.append(train_acc)
        self.print_log("\tMean train loss: {:.6f}".format(float(np.mean(loss_values))))
        self.print_log("\tTrain Accuracy: {:.2f}%".format(100 * train_acc))
        self.print_log("\t{}".format(self.class_accuracy_text(class_true, class_total)))

    def eval(self, epoch, loader_name=("test",), wrong_file=None, result_file=None):
        self.model.eval()
        f_w = open(wrong_file, "w") if wrong_file is not None else None
        f_r = open(result_file, "w") if result_file is not None else None

        self.print_log("Eval epoch: {}".format(epoch + 1))
        eval_results = dict()
        for ln in loader_name:
            num_class = int(self.arg.model_args.get("num_class", 4))
            class_total = np.zeros(num_class, dtype=np.int64)
            class_true = np.zeros(num_class, dtype=np.int64)
            total_correct = 0
            total_count = 0
            loss_values = []
            cls_values = []
            aff_values = []
            w_cls_values = []
            w_aff_values = []
            all_true = []
            all_pred = []

            process = tqdm(self.data_loader[ln])
            for batch_idx, (data_m, data_p, label, feature, index) in enumerate(process):
                with torch.no_grad():
                    data_m = data_m.float().to(self.device)
                    data_p = data_p.float().to(self.device)
                    label = label.long().to(self.device)
                    feature = feature.float().to(self.device)

                    output = self.model(data_p, data_m, label=label)
                    logits, loss, metrics = self.compute_losses(output, label, feature, epoch)

                    _, predict_label = torch.max(logits.data, 1)
                    total_correct += int((predict_label == label).sum().item())
                    total_count += label.size(0)

                    true_np = label.detach().cpu().numpy()
                    pred_np = predict_label.detach().cpu().numpy()
                    for pred, true in zip(pred_np, true_np):
                        class_total[true] += 1
                        class_true[true] += int(pred == true)
                    all_true.extend(true_np.tolist())
                    all_pred.extend(pred_np.tolist())

                    loss_values.append(loss.item())
                    cls_values.append(metrics["loss_cls"].item())
                    aff_values.append(metrics["loss_aff"].item())
                    w_cls_values.append(metrics["loss_weight_cls"].item())
                    w_aff_values.append(metrics["loss_weight_aff"].item())

                if f_r is not None or f_w is not None:
                    for i, pred in enumerate(pred_np):
                        true = int(true_np[i])
                        if f_r is not None:
                            f_r.write("{},{}\n".format(int(pred), true))
                        if int(pred) != true and f_w is not None:
                            f_w.write("{},{},{}\n".format(int(index[i]), int(pred), true))
                if self.arg.max_eval_batches > 0 and batch_idx + 1 >= self.arg.max_eval_batches:
                    break

            accuracy = total_correct * 1.0 / max(1, total_count)
            metric_values = self.classification_metrics(all_true, all_pred, num_class)
            metric_values.update(
                {
                    "epoch": epoch + 1,
                    "loader": ln,
                    "class_true": class_true.copy(),
                    "class_total": class_total.copy(),
                    "class_accuracy_text": self.class_accuracy_text(class_true, class_total),
                }
            )
            eval_results[ln] = metric_values
            self.last_eval_metrics[ln] = metric_values
            self.eval_acc_list.append(accuracy)
            if accuracy >= self.best_acc:
                self.best_acc = accuracy
                self.best_metrics = metric_values.copy()
                self.best_state_dict = self._capture_model_state()
                if self.arg.save_model:
                    torch.save(self.best_state_dict, self.best_ckpt_path)

            self.print_log("\tMean {} loss: {:.6f}".format(ln, float(np.mean(loss_values))))
            self.print_log("\tMean {} cls loss: {:.6f}".format(ln, float(np.mean(cls_values))))
            self.print_log("\tMean {} aff loss: {:.6f}".format(ln, float(np.mean(aff_values))))
            self.print_log(
                "\tMean {} loss weights: cls {:.4f}, aff {:.4f}".format(
                    ln,
                    float(np.mean(w_cls_values)),
                    float(np.mean(w_aff_values)),
                )
            )
            self.print_log("\tTop1: {:.2f}%".format(accuracy * 100))
            self.print_log("\tPrecision: {:.2f}%".format(metric_values["precision"] * 100))
            self.print_log("\tRecall: {:.2f}%".format(metric_values["recall"] * 100))
            self.print_log("\tF1: {:.4f}".format(metric_values["f1"]))
            self.print_log("\tG-mean: {:.2f}%".format(metric_values["gmean"] * 100))
            self.print_log("\tKappa: {:.4f}".format(metric_values["kappa"]))
            self.print_log("\tBest acc: {:.2f}%".format(self.best_acc * 100))
            self.print_log("\tSub-class Acc: {}".format(metric_values["class_accuracy_text"]))

        if f_w is not None:
            f_w.close()
        if f_r is not None:
            f_r.close()
        return eval_results

    def start(self):
        if self.arg.phase == "train":
            total_start = time.time()
            self.print_log("Parameters:\n{}\n".format(str(vars(self.arg))))
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if self.lr < 1e-6:
                    break
                start = time.time()
                self.train(epoch)
                self.print_log("Train epoch time: {:.2f}s".format(time.time() - start))

                start = time.time()
                self.eval(epoch, loader_name=("test",))
                self.print_log("Eval epoch time: {:.2f}s".format(time.time() - start))
            self.print_log("Best accuracy: {:.2f}%".format(self.best_acc * 100))
            self.print_final_summary(time.time() - total_start)
            if self.export_analysis_data:
                self.export_stage_analysis()
            if self.export_visuals:
                self.export_visualizations()
            if self.export_confusion:
                self.export_confusion_matrix()
        elif self.arg.phase == "test":
            if self.arg.weights is None:
                raise ValueError("Please appoint --weights.")
            total_start = time.time()
            wf = os.path.join(self.arg.model_saved_name, "wrong.txt")
            rf = os.path.join(self.arg.model_saved_name, "right.txt")
            self.arg.print_log = False
            self.print_log("Model:   {}.".format(self.arg.model))
            self.print_log("Weights: {}.".format(self.arg.weights))
            self.eval(epoch=0, loader_name=("test",), wrong_file=wf, result_file=rf)
            self.print_final_summary(time.time() - total_start)
            self.print_log("Done.")
        else:
            raise ValueError("Unsupported phase: {}".format(self.arg.phase))

    def close(self):
        self.train_writer.close()
        self.val_writer.close()


if __name__ == "__main__":
    parser = get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, "r") as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print("WRONG ARG: {}".format(k))
                assert k in key
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    try:
        processor.start()
    finally:
        processor.close()
