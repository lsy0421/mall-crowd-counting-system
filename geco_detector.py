"""
GeCo2 Few-shot 目标检测封装（商场人流等场景）。

依赖官方仓库中的模型与工具：请将 ``geco_root`` 指向 GECO2 源码根目录（内含 ``models/``、``utils/``）。
默认权重：``E:\\mallsys\\GECO2\\CNTQG_multitrain_ca44.pth``。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import torch
from torch.nn import DataParallel
from torchvision import transforms as T
from torchvision import ops

# ---------------------------------------------------------------------------
# 将 GECO2 源码根目录加入 sys.path，以便导入 models / utils
# ---------------------------------------------------------------------------
_DEFAULT_GECO_ROOT = Path(r"E:\mallsys\GECO2")
_DEFAULT_WEIGHT = Path(r"E:\mallsys\GECO2\CNTQG_multitrain_ca44.pth")


def _ensure_geco_on_path(geco_root: Path) -> Path:
    root = Path(geco_root).resolve()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def _import_geco_modules():
    from models.counter_infer import build_model
    from utils.arg_parser import get_argparser
    from utils.data import resize_and_pad

    return build_model, get_argparser, resize_and_pad


BoxLike = Union[Sequence[float], np.ndarray]
ExemplarItem = Union[
    BoxLike,
    dict,
    np.ndarray,
    torch.Tensor,
]


class GeCo2Detector:
    """
    基于 GeCo2（Few-shot）的检测器封装：纯 PyTorch 推理，输出统一为 numpy 检测矩阵。

    说明：官方实现中，范例通过「当前图像上的边界框」在 backbone 特征图上做 ROI Align，
    而不是在 ``detect`` 之外单独传入一张与待检测图无关的裁剪图文件路径。
    因此 ``set_exemplars`` 在仅提供裁剪图时，需要同时提供该裁剪在 **待检测帧** 像素坐标系下的 ``box``，
    或直接使用若干 ``[x1,y1,x2,y2]`` 框（与 ``detect(frame)`` 中 ``frame`` 尺寸一致）。
    """

    def __init__(
        self,
        weight_path: Union[str, Path] = _DEFAULT_WEIGHT,
        geco_root: Union[str, Path] = _DEFAULT_GECO_ROOT,
        threshold: float = 0.33,
        device: Union[str, torch.device, None] = None,
        infer_size: float | None = None,
    ) -> None:
        """
        :param weight_path: 官方权重 ``CNTQG_multitrain_ca44.pth`` 路径。
        :param geco_root: GECO2 仓库根目录（含 ``models``、``utils``）。
        :param threshold: 与官方 demo 一致，用于相对 ``box_v`` 最大值的筛选（见 ``geco2_engine``）。
        :param device: ``cuda`` / ``cpu``；默认自动选择。
        :param infer_size: 输入边长（与 ``args.image_size``、``resize_and_pad(..., size=...)`` 必须一致）。
        """
        self.geco_root = _ensure_geco_on_path(Path(geco_root))
        self.weight_path = Path(weight_path)
        self.threshold = float(threshold)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.infer_size = float(
            infer_size if infer_size is not None else os.getenv("GECO2_INFER_SIZE", "768")
        )

        self._build_model, self._get_argparser, self._resize_and_pad = _import_geco_modules()
        self.model = self._load_model()

        # 范例框：像素坐标 xyxy，与每次 detect 传入的 frame 同一坐标系、同一分辨率
        self._prompt_boxes: List[List[float]] | None = None

    def _load_model(self) -> DataParallel:
        if not self.weight_path.is_file():
            raise FileNotFoundError(f"未找到权重文件: {self.weight_path}")

        args = self._get_argparser().parse_args([])
        args.zero_shot = True
        args.image_size = int(self.infer_size)

        model = DataParallel(self._build_model(args).to(self.device))
        try:
            ckpt = torch.load(str(self.weight_path), map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(str(self.weight_path), map_location=self.device)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        # 实时推理：跳过 SAM mask 分支以省显存（与官方 geco2_engine 一致）
        model.module.return_masks = False
        model.module.skip_sam_mask = True
        return model

    def set_exemplars(self, exemplars_list: List[ExemplarItem]) -> None:
        """
        设置 Few-shot 范例 (仅处理图像框，不再处理零散的 Tensor)。
        """
        prompt_boxes: List[List[float]] = []
        self._use_precomputed_exemplar_features = False
        self._precomputed_feature_dict = None

        if not exemplars_list:
            raise ValueError("exemplars_list 不能为空。")

        for i, item in enumerate(exemplars_list):
            if isinstance(item, dict):
                box = item.get("box") or item.get("bbox")
                if box is None:
                    raise ValueError(f"范例 dict 必须包含 'box' 或 'bbox'（第 {i} 项）。")
                prompt_boxes.append(self._normalize_box(box))
                continue

            arr = np.asarray(item, dtype=np.float64)
            if arr.ndim == 1 and arr.size == 4:
                prompt_boxes.append([float(x) for x in arr])
                continue

            raise TypeError(f"不支持的常规范例类型（第 {i} 项）: {type(item)}，如需传入特征请使用 set_precomputed_features()")

        self._prompt_boxes = prompt_boxes

    def set_precomputed_features(self, feature_dict: dict, prompt_boxes: List[List[float]]) -> None:
        """
        直接注入预计算的特征字典及对应的原始框。
        :param feature_dict: 必须包含 'exemplars', 'exemplars_l1', 'exemplars_l2' 三个键
        :param prompt_boxes: 与特征对应的原始检测框，用于模型计算形状编码(Shape)
        """
        required_keys = {'exemplars', 'exemplars_l1', 'exemplars_l2'}
        if not required_keys.issubset(feature_dict.keys()):
            raise ValueError(f"feature_dict 必须包含这三个键: {required_keys}")
            
        self._precomputed_feature_dict = feature_dict
        self._prompt_boxes = prompt_boxes
        self._use_precomputed_exemplar_features = True

    @torch.no_grad()
    def extract_features(self, frame: np.ndarray, boxes: List[List[float]]) -> dict[str, torch.Tensor]:
        """
        在指定帧上按给定框提取范例 ROI 特征（与 ``models/counter_infer.py`` 中 ROI Align 分支一致），
        供 ``set_precomputed_features`` 锁死第一帧范例时使用。

        :param frame: BGR ``(H, W, 3)``，与 ``detect`` 使用同一套预处理。
        :param boxes: 范例框列表 ``[[x1,y1,x2,y2], ...]``，像素坐标，与 ``frame`` 尺寸一致。
        :return: 字典 ``exemplars`` / ``exemplars_l1`` / ``exemplars_l2``，形状均为
          ``[1, K * kernel_dim^2, emb_dim]``（``K=len(boxes)``），已 ``detach().cpu()`` 便于持久化。
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame 须为 (H,W,3) BGR，当前 shape={frame.shape}")
        if not boxes:
            raise ValueError("boxes 不能为空。")

        image_rgb = frame[:, :, ::-1].copy()
        image_tensor = torch.tensor(image_rgb, device=self.device)
        image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0
        image_tensor = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(image_tensor)

        boxes_t = torch.tensor(boxes, dtype=torch.float32, device=self.device)
        if boxes_t.ndim != 2 or boxes_t.shape[1] != 4:
            raise ValueError(f"boxes 须为 Kx4 的 xyxy，当前 shape={boxes_t.shape}")

        img, bboxes, _scale = self._resize_and_pad(
            image_tensor, boxes_t, size=self.infer_size
        )
        x = img.unsqueeze(0).to(self.device)
        bboxes_b = bboxes.unsqueeze(0).to(self.device)

        cnt = self.model.module
        feats = cnt.backbone(x)

        src = feats["vision_features"]
        bs, _c, w, _h = src.shape
        reduction = float(cnt.image_size) / float(w)

        num_objects = bboxes_b.size(1)
        kernel_dim = 1

        bboxes_roi = torch.cat(
            [
                torch.arange(bs, device=bboxes_b.device)
                .repeat_interleave(num_objects)
                .reshape(-1, 1)
                .to(dtype=bboxes_b.dtype),
                bboxes_b.flatten(0, 1),
            ],
            dim=1,
        )

        emb_dim = cnt.emb_dim

        exemplars = (
            ops.roi_align(
                src,
                bboxes_roi,
                output_size=(kernel_dim, kernel_dim),
                spatial_scale=1.0 / reduction,
                aligned=True,
            )
            .permute(0, 2, 3, 1)
            .reshape(bs, num_objects * kernel_dim**2, emb_dim)
        )

        l1 = feats["backbone_fpn"][0]
        l2 = feats["backbone_fpn"][1]

        exemplars_l1 = (
            ops.roi_align(
                l1,
                bboxes_roi,
                output_size=(kernel_dim, kernel_dim),
                spatial_scale=1.0 / reduction * 2.0 * 2.0,
                aligned=True,
            )
            .permute(0, 2, 3, 1)
            .reshape(bs, num_objects * kernel_dim**2, emb_dim)
        )

        exemplars_l2 = (
            ops.roi_align(
                l2,
                bboxes_roi,
                output_size=(kernel_dim, kernel_dim),
                spatial_scale=1.0 / reduction * 2.0,
                aligned=True,
            )
            .permute(0, 2, 3, 1)
            .reshape(bs, num_objects * kernel_dim**2, emb_dim)
        )

        return {
            "exemplars": exemplars.detach().cpu(),
            "exemplars_l1": exemplars_l1.detach().cpu(),
            "exemplars_l2": exemplars_l2.detach().cpu(),
        }

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        对单帧 BGR 图像做前向推理。
        """
        if getattr(self, "_prompt_boxes", None) is None:
            raise ValueError("请先调用 set_exemplars(...) 或 set_precomputed_features(...)")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame 须为 (H,W,3) BGR，当前 shape={frame.shape}")

        image_rgb = frame[:, :, ::-1].copy()
        image_tensor = torch.tensor(image_rgb, device=self.device)
        image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0
        image_tensor = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(image_tensor)

        real_boxes_tensor = torch.tensor(
            self._prompt_boxes, dtype=torch.float32, device=self.device
        )

        img, bboxes, scale = self._resize_and_pad(
            image_tensor, real_boxes_tensor, size=self.infer_size
        )
        img = img.unsqueeze(0).to(self.device)
        
        # 无论哪种分支，bboxes 都要增加 batch 维度，并在相同设备上
        bboxes = bboxes.unsqueeze(0).to(self.device)
        
        # 分支判断：是用预计算的特征字典，还是走原图特征提取
        if getattr(self, "_use_precomputed_exemplar_features", False):
            # 将字典中的 tensor 移动到正确的计算设备上
            feat_dict = {k: v.to(self.device) for k, v in self._precomputed_feature_dict.items()}
            # 必须同时传入 bboxes 和 exemplar_features
            outputs, _, _, _, _ = self.model(img, bboxes, exemplar_features=feat_dict)
        else:
            outputs, _, _, _, _ = self.model(img, bboxes)

        idx = 0
        thr_inv = 1.0 / self.threshold

        pred_boxes = outputs[idx]["pred_boxes"]
        box_v = outputs[idx]["box_v"]

        if pred_boxes.dim() == 3 and pred_boxes.size(0) == 1:
            pred_boxes = pred_boxes[0]
        if box_v.dim() == 2 and box_v.size(0) == 1:
            box_v = box_v[0]

        sel = box_v > (box_v.max() / thr_inv)
        if sel.sum().item() == 0:
            return np.zeros((0, 6), dtype=np.float32)

        keep = ops.nms(pred_boxes[sel], box_v[sel], 0.5)
        final_boxes = pred_boxes[sel][keep]
        final_scores = box_v[sel][keep]

        final_boxes = torch.clamp(final_boxes, 0, 1)
        final_boxes = (final_boxes / scale * img.shape[-1]).cpu()
        final_scores = final_scores.cpu()

        n = final_boxes.shape[0]
        out = np.zeros((n, 6), dtype=np.float32)
        out[:, :4] = final_boxes.numpy().astype(np.float32)
        out[:, 4] = final_scores.numpy().astype(np.float32)
        out[:, 5] = 0.0
        return out