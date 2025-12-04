# ------------------------------------------------------------------
# SimpleNet: A Simple Network for Image Anomaly Detection and Localization (https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization_CVPR_2023_paper.pdf)
# Github source: https://github.com/DonaldRR/SimpleNet
# Licensed under the MIT License [see LICENSE for details]
# The script is based on the code of PatchCore (https://github.com/amazon-science/patchcore-inspection)
# ------------------------------------------------------------------

"""detection methods."""
import logging
import os
import pickle
from collections import OrderedDict
from pathlib import Path

import math
import numpy as np
import torch
import torch.nn.functional as F
import time
import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import backbones
import common
import metrics
from utils import plot_segmentation_images

LOGGER = logging.getLogger(__name__)

def init_weight(m):

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class ResidualBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.BatchNorm1d(dim),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.2),
        )
        self.block.apply(init_weight)

    def forward(self, x):
        return x + self.block(x)


class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        modules = [
            torch.nn.Linear(in_planes, _hidden),
            torch.nn.BatchNorm1d(_hidden),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.2),
            ResidualBlock(_hidden),
        ]

        # If more layers are requested, stack additional residual blocks.
        for _ in range(max(n_layers - 1, 0)):
            modules.append(ResidualBlock(_hidden))

        self.body = torch.nn.Sequential(*modules)
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x


class Projection(torch.nn.Module):
    
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()
        
        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes 
            self.layers.add_module(f"{i}fc", 
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                # if layer_type > 0:
                #     self.layers.add_module(f"{i}bn", 
                #                            torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)
    
    def forward(self, x):
        
        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x


class TBWrapper:
    
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)
    
    def step(self):
        self.g_iter += 1

class SimpleNet(torch.nn.Module):
    def __init__(self, device):
        """anomaly detection class."""
        super(SimpleNet, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension, # 1536
        target_embed_dimension, # 1536
        patchsize=3, # 3
        patchstride=1, 
        embedding_size=None, # 256
        meta_epochs=1, # 40
        aed_meta_epochs=1,
        gan_epochs=1, # 4
        noise_std=0.05,
        mix_noise=1,
        noise_type="GAU",
        dsc_layers=2, # 2
        dsc_hidden=None, # 1024
        dsc_margin=.8, # .5
        dsc_lr=0.0002,
        train_backbone=False,
        auto_noise=0,
        cos_lr=False,
        lr=1e-3,
        pre_proj=0, # 1
        proj_layer_type=0,
        use_distillation=False,
        teacher_backbone="wideresnet50",
        distill_weight=0.1,
        distill_max_patches=192,
        distill_knn=5,
        distill_direction_weight=1.0,
        distill_distance_weight=1.0,
        distill_neighborhood_weight=1.0,
        distill_embedding_weight=0.5,
        distill_tightness_weight=0.25,
        **kwargs,
    ):
        pid = os.getpid()
        def show_mem():
            return(psutil.Process(pid).memory_info())

        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.embedding_size = embedding_size if embedding_size is not None else self.target_embed_dimension
        self.meta_epochs = meta_epochs
        self.lr = lr
        self.cos_lr = cos_lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)
        # AED
        self.aed_meta_epochs = aed_meta_epochs

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj, proj_layer_type)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.AdamW(self.pre_projection.parameters(), lr*.1)

        # Discriminator
        self.auto_noise = [auto_noise, None]
        self.dsc_lr = dsc_lr
        self.gan_epochs = gan_epochs
        self.mix_noise = mix_noise
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.dsc_lr, weight_decay=1e-5)
        self.dsc_schl = torch.optim.lr_scheduler.CosineAnnealingLR(self.dsc_opt, (meta_epochs - aed_meta_epochs) * gan_epochs, self.dsc_lr*.4)
        self.dsc_margin= dsc_margin

        # Distillation controls
        self.use_distillation = use_distillation
        self.distill_weight = distill_weight
        self.distill_max_patches = distill_max_patches
        self.distill_knn = distill_knn
        self.distill_direction_weight = distill_direction_weight
        self.distill_distance_weight = distill_distance_weight
        self.distill_neighborhood_weight = distill_neighborhood_weight
        self.distill_embedding_weight = distill_embedding_weight
        self.distill_tightness_weight = distill_tightness_weight
        if self.use_distillation:
            self.teacher_backbone = backbones.load(teacher_backbone)
            self.teacher_backbone.name = teacher_backbone
            self.teacher_backbone.eval()
            for p in self.teacher_backbone.parameters():
                p.requires_grad = False
            teacher_feature_aggregator = common.NetworkFeatureAggregator(
                self.teacher_backbone, self.layers_to_extract_from, self.device, False
            )
            teacher_feature_dimensions = teacher_feature_aggregator.feature_dimensions(input_shape)
            teacher_preprocessing = common.Preprocessing(
                teacher_feature_dimensions, pretrain_embed_dimension
            )
            teacher_preadapt_aggregator = common.Aggregator(
                target_dim=target_embed_dimension
            )
            _ = teacher_preadapt_aggregator.to(self.device)
            self.teacher_modules = torch.nn.ModuleDict({
                "feature_aggregator": teacher_feature_aggregator,
                "preprocessing": teacher_preprocessing,
                "preadapt_aggregator": teacher_preadapt_aggregator,
            })
            self.teacher_modules.to(self.device)

        self.model_dir = ""
        self.dataset_name = ""
        self.tau = 1
        self.logger = None

    def set_model_dir(self, model_dir, dataset_name):

        self.model_dir = model_dir 
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir) #SummaryWriter(log_dir=tb_dir)
    

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                    input_image = image.to(torch.float).to(self.device)
                with torch.no_grad():
                    features.append(self._embed(input_image)[0])
            return features
        return self._embed(data)[0]

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False, return_layer_features=False, modules=None):
        """Returns feature embeddings for images."""

        modules = modules if modules is not None else self.forward_modules
        B = len(images)
        if not evaluation and self.train_backbone:
            modules["feature_aggregator"].train()
            features = modules["feature_aggregator"](images, eval=evaluation)
        else:
            _ = modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        layer_features = [x if return_layer_features else None for x in features]
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = modules["preprocessing"](features) # pooling each feature to same channel and stack together
        features = modules["preadapt_aggregator"](features) # further pooling

        return features, patch_shapes, layer_features

    
    def test(
        self,
        training_data,
        test_data,
        save_segmentation_images,
        visualize_report=False,
        report_dir="analysis",
        report_sample_index=0,
    ):

        ckpt_candidates = ["ckpt.pth", "models.ckpt"]
        state_dicts = None
        for candidate in ckpt_candidates:
            ckpt_path = os.path.join(self.ckpt_dir, candidate)
            if os.path.exists(ckpt_path):
                state_dicts = torch.load(ckpt_path, map_location=self.device)
                break

        if state_dicts:
            if "pretrained_enc" in state_dicts:
                self.feature_enc.load_state_dict(state_dicts["pretrained_enc"])
            if "pretrained_dec" in state_dicts:
                self.feature_dec.load_state_dict(state_dicts["pretrained_dec"])
            if "discriminator" in state_dicts:
                self.discriminator.load_state_dict(state_dicts["discriminator"])
            if "pre_projection" in state_dicts and self.pre_proj > 0:
                self.pre_projection.load_state_dict(state_dicts["pre_projection"])

        (
            scores,
            segmentations,
            features,
            _labels_gt,
            masks_gt,
            inference_time,
        ) = self.predict(test_data)

        anomaly_labels = [
            x[1] != "good" for x in test_data.dataset.data_to_iterate
        ]

        if save_segmentation_images:
            self.save_segmentation_images(test_data, segmentations, scores)

        auroc, full_pixel_auroc, pro, inference_time, norm_segmentations, norm_scores = self._evaluate(
            test_data,
            scores,
            segmentations,
            features,
            anomaly_labels,
            masks_gt,
            inference_time,
        )

        if visualize_report:
            self._visualize_predictions(
                test_data,
                norm_scores,
                norm_segmentations,
                anomaly_labels,
                masks_gt,
                report_dir,
                report_sample_index,
            )

        LOGGER.info(
            f"Average inference time per image: {inference_time:.6f} seconds"
        )

        return auroc, full_pixel_auroc, pro, inference_time

    def _evaluate(self, test_data, scores, segmentations, features, labels_gt, masks_gt, inference_time=None):

        scores = np.squeeze(np.array(scores))
        norm_scores = self._normalize_scores(scores)

        auroc = metrics.compute_imagewise_retrieval_metrics(
            norm_scores, labels_gt
        )["auroc"]

        if len(masks_gt) > 0:
            norm_segmentations = self._normalize_segmentations(segmentations)

            # Compute PRO score & PW Auroc for all images
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                norm_segmentations, masks_gt)
            full_pixel_auroc = pixel_scores["auroc"]

            pro = metrics.compute_pro(np.array(masks_gt), norm_segmentations)
        else:
            full_pixel_auroc = -1
            pro = -1
            norm_segmentations = np.array(segmentations)

        return auroc, full_pixel_auroc, pro, inference_time, norm_segmentations, norm_scores

    @staticmethod
    def _normalize_segmentations(segmentations):
        segmentations = np.array(segmentations)
        min_scores = (
            segmentations.reshape(len(segmentations), -1)
            .min(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        max_scores = (
            segmentations.reshape(len(segmentations), -1)
            .max(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        norm_segmentations = []
        for seg, min_score, max_score in zip(segmentations, min_scores, max_scores):
            norm_segmentations.append((seg - min_score) / max(max_score - min_score, 1e-8))
        return np.stack(norm_segmentations)

    @staticmethod
    def _normalize_scores(scores):
        img_min_scores = scores.min(axis=-1)
        img_max_scores = scores.max(axis=-1)
        return (scores - img_min_scores) / (img_max_scores - img_min_scores + 1e-8)

    def _visualize_predictions(
        self,
        test_loader,
        norm_scores,
        norm_segmentations,
        labels_gt,
        masks_gt,
        report_dir,
        report_sample_index,
    ):
        report_root = Path(report_dir)
        report_root.mkdir(parents=True, exist_ok=True)

        sample_index = int(report_sample_index) % len(norm_segmentations)
        dataset = test_loader.dataset
        sample = dataset[sample_index]

        image = sample["image"].cpu().numpy()
        mean = np.array(dataset.transform_mean).reshape(3, 1, 1)
        std = np.array(dataset.transform_std).reshape(3, 1, 1)
        image_uint8 = np.clip((image * std + mean) * 255, 0, 255).astype(np.uint8)
        image_uint8 = np.transpose(image_uint8, (1, 2, 0))

        heatmap = np.squeeze(norm_segmentations[sample_index])
        pixel_threshold = float(np.median(norm_segmentations))
        pred_mask = (heatmap > pixel_threshold).astype(float)
        gt_mask = sample.get("mask", np.zeros_like(pred_mask))
        gt_mask = np.squeeze(gt_mask).cpu().numpy()

        plt.figure(figsize=(6, 6))
        plt.imshow(image_uint8)
        plt.imshow(heatmap.squeeze(), cmap="jet", alpha=0.5)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(report_root / "sample_heatmap_overlay.png", dpi=200)
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.imshow(image_uint8)
        plt.imshow(gt_mask, cmap="Greens", alpha=0.35)
        plt.imshow(pred_mask, cmap="Reds", alpha=0.35)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(report_root / "prediction_vs_gt.png", dpi=200)
        plt.close()

        flat_masks = np.array(masks_gt)
        if flat_masks.size == 0:
            return

        pixel_scores = norm_segmentations.reshape(len(norm_segmentations), -1)
        pixel_labels = flat_masks.reshape(len(flat_masks), -1)
        negative_pixels = (pixel_labels == 0).sum(axis=1)
        fp_counts = np.logical_and(pixel_scores > pixel_threshold, pixel_labels == 0).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            fpr = np.divide(fp_counts, negative_pixels, where=negative_pixels > 0)

        plt.figure(figsize=(7, 4))
        plt.hist(fpr, bins=20, color="#ff7f50", edgecolor="black")
        plt.xlabel("False positive rate per image")
        plt.ylabel("Count")
        plt.title("Pixel-level FPR distribution")
        plt.tight_layout()
        plt.savefig(report_root / "fpr_distribution.png", dpi=200)
        plt.close()

        image_level_scores = norm_scores
        if image_level_scores.ndim > 1:
            image_level_scores = image_level_scores.reshape(len(image_level_scores), -1).max(axis=1)
        image_threshold = float(np.median(image_level_scores))
        image_predictions = image_level_scores > image_threshold

        false_positives = []
        false_negatives = []
        image_entries = test_loader.dataset.data_to_iterate
        for idx, (pred, label) in enumerate(zip(image_predictions, labels_gt)):
            if pred and label == 0:
                false_positives.append(image_entries[idx][2])
            if (not pred) and label == 1:
                false_negatives.append(image_entries[idx][2])

        plt.figure(figsize=(7, 4))
        plt.hist(image_level_scores, bins=30, color="#4c72b0", edgecolor="black")
        plt.axvline(image_threshold, color="red", linestyle="--", label=f"Median threshold: {image_threshold:.3f}")
        plt.xlabel("Normalized image anomaly score")
        plt.ylabel("Count")
        plt.title("Image-level score distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(report_root / "image_score_distribution.png", dpi=200)
        plt.close()

        summary = {
            "sample_index": sample_index,
            "pixel_threshold_median": pixel_threshold,
            "image_threshold_median": image_threshold,
            "false_positive_images": false_positives,
            "false_negative_images": false_negatives,
        }
        summary_path = report_root / "visual_report.txt"
        with open(summary_path, "w", encoding="utf-8") as handle:
            for key, value in summary.items():
                handle.write(f"{key}: {value}\n")

        
    
    def train(self, training_data, test_data):

        
        state_dict = {}
        ckpt_path = os.path.join(self.ckpt_dir, "ckpt.pth")
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)

            self.predict(training_data, "train_")
            scores, segmentations, features, labels_gt, masks_gt, inference_time = self.predict(test_data)
            (
                auroc,
                full_pixel_auroc,
                anomaly_pixel_auroc,
                inference_time,
                _norm_segmentations,
                _norm_scores,
            ) = self._evaluate(
                test_data, scores, segmentations, features, labels_gt, masks_gt, inference_time
            )

            LOGGER.info(f"Average inference time per image: {inference_time:.6f} seconds")

            return auroc, full_pixel_auroc, anomaly_pixel_auroc, inference_time
        
        def update_state_dict(d):
            
            state_dict["discriminator"] = OrderedDict({
                k:v.detach().cpu() 
                for k, v in self.discriminator.state_dict().items()})
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k:v.detach().cpu() 
                    for k, v in self.pre_projection.state_dict().items()})

        best_record = None
        for i_mepoch in range(self.meta_epochs):

            self._train_discriminator(training_data)

            # torch.cuda.empty_cache()
            scores, segmentations, features, labels_gt, masks_gt, inference_time = self.predict(test_data)
            (
                auroc,
                full_pixel_auroc,
                pro,
                inference_time,
                _norm_segmentations,
                _norm_scores,
            ) = self._evaluate(
                test_data, scores, segmentations, features, labels_gt, masks_gt, inference_time
            )
            self.logger.logger.add_scalar("i-auroc", auroc, i_mepoch)
            self.logger.logger.add_scalar("p-auroc", full_pixel_auroc, i_mepoch)
            self.logger.logger.add_scalar("pro", pro, i_mepoch)

            if best_record is None:
                best_record = [auroc, full_pixel_auroc, pro, inference_time]
                update_state_dict(state_dict)
                # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})
            else:
                if auroc > best_record[0]:
                    best_record = [auroc, full_pixel_auroc, pro, inference_time]
                    update_state_dict(state_dict)
                    # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})
                elif auroc == best_record[0] and full_pixel_auroc > best_record[1]:
                    best_record[1] = full_pixel_auroc
                    best_record[2] = pro
                    best_record[3] = inference_time
                    update_state_dict(state_dict)
                    # state_dict = OrderedDict({k:v.detach().cpu() for k, v in self.state_dict().items()})

            print(f"----- {i_mepoch} I-AUROC:{round(auroc, 4)}(MAX:{round(best_record[0], 4)})"
                  f"  P-AUROC{round(full_pixel_auroc, 4)}(MAX:{round(best_record[1], 4)}) -----"
                  f"  PRO-AUROC{round(pro, 4)}(MAX:{round(best_record[2], 4)}) -----"
                  f"  INF-TIME:{round(inference_time, 6)}(BEST:{round(best_record[3], 6)}) -----")
        
        torch.save(state_dict, ckpt_path)
        
        return best_record
            

    def _train_discriminator(self, input_data):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()
        
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()
        # self.feature_enc.eval()
        # self.feature_dec.eval()
        i_iter = 0
        LOGGER.info(f"Training discriminator...")
        with tqdm.tqdm(total=self.gan_epochs) as pbar:
            for i_epoch in range(self.gan_epochs):
                all_loss = []
                all_p_true = []
                all_p_fake = []
                all_p_interp = []
                embeddings_list = []
                for data_item in input_data:
                    self.dsc_opt.zero_grad()
                    if self.pre_proj > 0:
                        self.proj_opt.zero_grad()
                    # self.dec_opt.zero_grad()

                    i_iter += 1
                    img = data_item["image"]
                    img = img.to(torch.float).to(self.device)
                    student_feats, _, student_layer_feats = self._embed(
                        img,
                        evaluation=False,
                        return_layer_features=self.use_distillation,
                    )
                    if self.pre_proj > 0:
                        true_feats = self.pre_projection(student_feats)
                    else:
                        true_feats = student_feats
                    
                    noise_idxs = torch.randint(0, self.mix_noise, torch.Size([true_feats.shape[0]]))
                    noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=self.mix_noise).to(self.device) # (N, K)
                    noise = torch.stack([
                        torch.normal(0, self.noise_std * 1.1**(k), true_feats.shape)
                        for k in range(self.mix_noise)], dim=1).to(self.device) # (N, K, C)
                    noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
                    fake_feats = true_feats + noise

                    scores = self.discriminator(torch.cat([true_feats, fake_feats]))
                    true_scores = scores[:len(true_feats)]
                    fake_scores = scores[len(fake_feats):]
                    
                    th = self.dsc_margin
                    p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                    p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
                    true_loss = torch.clip(-true_scores + th, min=0)
                    fake_loss = torch.clip(fake_scores + th, min=0)

                    self.logger.logger.add_scalar(f"p_true", p_true, self.logger.g_iter)
                    self.logger.logger.add_scalar(f"p_fake", p_fake, self.logger.g_iter)

                    loss = true_loss.mean() + fake_loss.mean()
                    if self.use_distillation:
                        with torch.no_grad():
                            teacher_feats, _, teacher_layer_feats = self._embed(
                                img,
                                evaluation=True,
                                return_layer_features=True,
                                modules=self.teacher_modules,
                            )
                        distill_loss = self._manifold_distillation_loss(
                            student_layer_feats,
                            teacher_layer_feats,
                            true_feats,
                            teacher_feats,
                        )
                        loss = loss + self.distill_weight * distill_loss
                        self.logger.logger.add_scalar("distill/loss", distill_loss, self.logger.g_iter)
                    self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
                    self.logger.step()

                    loss.backward()
                    if self.pre_proj > 0:
                        self.proj_opt.step()
                    if self.train_backbone:
                        self.backbone_opt.step()
                    self.dsc_opt.step()

                    loss = loss.detach().cpu() 
                    all_loss.append(loss.item())
                    all_p_true.append(p_true.cpu().item())
                    all_p_fake.append(p_fake.cpu().item())
                
                if len(embeddings_list) > 0:
                    self.auto_noise[1] = torch.cat(embeddings_list).std(0).mean(-1)
                
                if self.cos_lr:
                    self.dsc_schl.step()
                
                all_loss = sum(all_loss) / len(input_data)
                all_p_true = sum(all_p_true) / len(input_data)
                all_p_fake = sum(all_p_fake) / len(input_data)
                cur_lr = self.dsc_opt.state_dict()['param_groups'][0]['lr']
                pbar_str = f"epoch:{i_epoch} loss:{round(all_loss, 5)} "
                pbar_str += f"lr:{round(cur_lr, 6)}"
                pbar_str += f" p_true:{round(all_p_true, 3)} p_fake:{round(all_p_fake, 3)}"
                if len(all_p_interp) > 0:
                    pbar_str += f" p_interp:{round(sum(all_p_interp) / len(input_data), 3)}"
                pbar.set_description_str(pbar_str)
                pbar.update(1)


    def _manifold_distillation_loss(self, student_layers, teacher_layers, student_embed, teacher_embed):
        """Align manifold structures between teacher and student."""

        loss_total = torch.tensor(0.0, device=self.device)
        eps = 1e-6

        if student_layers is not None and teacher_layers is not None:
            direction_losses = []
            distance_losses = []
            neighborhood_losses = []
            for s_feat, t_feat in zip(student_layers, teacher_layers):
                if s_feat is None or t_feat is None:
                    continue
                s_tokens = s_feat.flatten(2)
                t_tokens = t_feat.flatten(2)
                patch_count = min(s_tokens.shape[1], t_tokens.shape[1])
                if patch_count > self.distill_max_patches:
                    idx = torch.randperm(patch_count, device=self.device)[: self.distill_max_patches]
                    s_tokens = s_tokens[:, idx]
                    t_tokens = t_tokens[:, idx]
                else:
                    s_tokens = s_tokens[:, :patch_count]
                    t_tokens = t_tokens[:, :patch_count]

                s_tokens = F.normalize(s_tokens, dim=-1)
                t_tokens = F.normalize(t_tokens, dim=-1)

                direction_losses.append((1 - (s_tokens * t_tokens).sum(-1)).mean())

                s_dist = torch.cdist(s_tokens, s_tokens) + eps
                t_dist = torch.cdist(t_tokens, t_tokens) + eps
                s_dist = s_dist / s_dist.mean(dim=(1, 2), keepdim=True)
                t_dist = t_dist / t_dist.mean(dim=(1, 2), keepdim=True)
                distance_losses.append(F.smooth_l1_loss(s_dist, t_dist))

                student_rel = torch.matmul(s_tokens, s_tokens.transpose(1, 2))
                teacher_rel = torch.matmul(t_tokens, t_tokens.transpose(1, 2))
                knn = min(self.distill_knn, teacher_rel.shape[-1] - 1)
                if knn > 0:
                    _, knn_idx = torch.topk(teacher_rel, k=knn + 1, dim=-1)
                    knn_idx = knn_idx[:, :, 1:]
                    teacher_neighbors = torch.gather(teacher_rel, 2, knn_idx)
                    student_neighbors = torch.gather(student_rel, 2, knn_idx)
                    neighborhood_losses.append(F.smooth_l1_loss(student_neighbors, teacher_neighbors))

            if direction_losses:
                loss_total = loss_total + self.distill_direction_weight * sum(direction_losses) / len(direction_losses)
            if distance_losses:
                loss_total = loss_total + self.distill_distance_weight * sum(distance_losses) / len(distance_losses)
            if neighborhood_losses:
                loss_total = loss_total + self.distill_neighborhood_weight * sum(neighborhood_losses) / len(neighborhood_losses)

        if student_embed is not None and teacher_embed is not None and student_embed.shape[0] > 1:
            s_norm = F.normalize(student_embed, dim=-1)
            t_norm = F.normalize(teacher_embed, dim=-1)
            s_pair = torch.cdist(s_norm, s_norm) + eps
            t_pair = torch.cdist(t_norm, t_norm) + eps
            s_pair = s_pair / s_pair.mean()
            t_pair = t_pair / t_pair.mean()
            embed_shape_loss = F.smooth_l1_loss(s_pair, t_pair)
            loss_total = loss_total + self.distill_embedding_weight * embed_shape_loss

            s_dispersion = s_norm.std(dim=0).mean()
            t_dispersion = t_norm.std(dim=0).mean()
            tightness_loss = torch.relu(s_dispersion - t_dispersion)
            loss_total = loss_total + self.distill_tightness_weight * tightness_loss

        return loss_total


    def predict(self, data, prefix=""):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, prefix)
        return self._predict(data)

    def _predict_dataloader(self, dataloader, prefix):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()


        img_paths = []
        scores = []
        masks = []
        features = []
        labels_gt = []
        masks_gt = []
        from sklearn.manifold import TSNE

        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            start_time = time.perf_counter()
            total_images = 0
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask", None) is not None:
                        masks_gt.extend(data["mask"].numpy().tolist())
                    image = data["image"]
                    img_paths.extend(data['image_path'])
                _scores, _masks, _feats = self._predict(image)
                for score, mask, feat, is_anomaly in zip(_scores, _masks, _feats, data["is_anomaly"].numpy().tolist()):
                    scores.append(score)
                    masks.append(mask)
                total_images += image.shape[0]

        inference_time = time.perf_counter() - start_time
        avg_inference_time = inference_time / max(total_images, 1)

        return scores, masks, features, labels_gt, masks_gt, avg_inference_time

    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()
        with torch.no_grad():
            features, patch_shapes, _ = self._embed(
                images,
                provide_patch_shapes=True,
                evaluation=True,
            )
            if self.pre_proj > 0:
                features = self.pre_projection(features)

            # features = features.cpu().numpy()
            # features = np.ascontiguousarray(features.cpu().numpy())
            patch_scores = image_scores = -self.discriminator(features)
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            features = features.reshape(batchsize, scales[0], scales[1], -1)
            masks, features = self.anomaly_segmentor.convert_to_segmentation(patch_scores, features)

        return list(image_scores), list(masks), list(features)

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "params.pkl")

    def save_to_path(self, save_path: str, prepend: str = ""):
        LOGGER.info("Saving data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(params, save_file, pickle.HIGHEST_PROTOCOL)

    def save_segmentation_images(self, data, segmentations, scores):
        image_paths = [
            x[2] for x in data.dataset.data_to_iterate
        ]
        mask_paths = [
            x[3] for x in data.dataset.data_to_iterate
        ]

        def image_transform(image):
            in_std = np.array(
                data.dataset.transform_std
            ).reshape(-1, 1, 1)
            in_mean = np.array(
                data.dataset.transform_mean
            ).reshape(-1, 1, 1)
            image = data.dataset.transform_img(image)
            return np.clip(
                (image.numpy() * in_std + in_mean) * 255, 0, 255
            ).astype(np.uint8)

        def mask_transform(mask):
            return data.dataset.transform_mask(mask).numpy()

        plot_segmentation_images(
            './output',
            image_paths,
            segmentations,
            scores,
            mask_paths,
            image_transform=image_transform,
            mask_transform=mask_transform,
        )

# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x
