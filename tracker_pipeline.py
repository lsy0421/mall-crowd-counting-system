"""
视频流预处理 + ByteTrack 追踪管线。

依赖: ``pip install supervision``（内置 ByteTrack 实现；底层关联匹配常用 ``lap``，由 supervision 依赖解析）。

与 ``geco_detector.GeCo2Detector.detect`` 输出对接：``dets`` 为 ``(N, 6)``，
列依次为 ``x1, y1, x2, y2, confidence, class_id``。
"""

from __future__ import annotations

import inspect
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

try:
    import supervision as sv
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "请先安装 supervision（提供 ByteTrack 封装）: pip install supervision"
    ) from exc


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    画面预处理：高斯滤波去噪 + CLAHE（在 LAB 的 L 通道）做逆光/对比度补偿。

    :param frame: OpenCV BGR 图像，``uint8``，形状 ``(H, W, 3)``。
    :return: 同尺寸、同类型的 BGR 图像。
    """
    if frame is None or frame.size == 0:
        raise ValueError("frame 不能为空")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame 须为 HxWx3 BGR，当前 shape={frame.shape}")

    # 高斯滤波抑制高频噪声（核尺寸须为正奇数）
    denoised = cv2.GaussianBlur(frame, ksize=(5, 5), sigmaX=0)

    # CLAHE 作用于亮度通道，避免直接对 BGR 三通道均衡导致色偏
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    merged = cv2.merge([l_eq, a_ch, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


class TrackerManager:
    """
    封装 ``supervision.ByteTrack``，并维护每个 ``track_id`` 的首次/最近活跃时间戳。
    """

    def __init__(
        self,
        frame_rate: int = 30,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        **byte_track_kwargs: Any,
    ) -> None:
        """
        :param frame_rate: 视频帧率，传入 ByteTrack（影响轨迹生命周期等内部参数）。
        :param track_activation_threshold: 高置信检测进入第一轮关联的阈值。
        :param lost_track_buffer: 丢失后保留轨迹的帧数。
        :param minimum_matching_threshold: IoU 匹配阈值。
        :param byte_track_kwargs: 其他传给 ``sv.ByteTrack`` 的关键字参数（若版本支持）。
        """
        self.track_history: Dict[int, Dict[str, float]] = {}
        kwargs: Dict[str, Any] = {
            "frame_rate": frame_rate,
            "track_activation_threshold": track_activation_threshold,
            "lost_track_buffer": lost_track_buffer,
            "minimum_matching_threshold": minimum_matching_threshold,
        }
        kwargs.update(byte_track_kwargs)
        # 不同版本 supervision 可能增减参数，做一次过滤以避免 TypeError
        params = inspect.signature(sv.ByteTrack.__init__).parameters
        filtered = {k: v for k, v in kwargs.items() if k in params}
        self._tracker = sv.ByteTrack(**filtered)

    def _empty_detections(self) -> "sv.Detections":
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=np.int32),
        )

    def update(self, dets: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        将 GeCo2（或其它检测器）的输出送入 ByteTrack，并更新 ``track_history``。

        :param dets: ``numpy.ndarray``，形状 ``(N, 6)``，每行
          ``[x1, y1, x2, y2, confidence, class_id]``（与 ``GeCo2Detector.detect`` 一致）。
        :param frame: 与当前帧对应的 BGR 图像；用于接口一致性及可选校验（ByteTrack 本身仅依赖 ``dets``）。
        :return: ``numpy.ndarray``，形状 ``(M, 5)``，每行 ``[x1, y1, x2, y2, track_id]``（``float32``，
          ``track_id`` 取整数值）。未匹配到轨迹的检测可能被过滤，取决于 supervision 行为。
        """
        if frame is None or frame.size == 0:
            raise ValueError("frame 不能为空")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame 须为 HxWx3 BGR，当前 shape={frame.shape}")

        dets = np.asarray(dets, dtype=np.float32)
        if dets.size == 0:
            dets = np.zeros((0, 6), dtype=np.float32)
        elif dets.ndim != 2 or dets.shape[1] != 6:
            raise ValueError(
                "dets 须为 (N, 6)，列: x1,y1,x2,y2,conf,cls；空检测可为 (0, 6) 或空列表"
            )

        now = time.time()

        if dets.shape[0] == 0:
            detections = self._empty_detections()
        else:
            detections = sv.Detections(
                xyxy=dets[:, :4].astype(np.float32),
                confidence=dets[:, 4].astype(np.float32),
                class_id=np.rint(dets[:, 5]).astype(np.int32),
            )

        tracked = self._tracker.update_with_detections(detections)

        if len(tracked) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        xyxy = tracked.xyxy
        tids = tracked.tracker_id
        if tids is None:
            return np.zeros((0, 5), dtype=np.float32)

        tids = np.asarray(tids)
        # 过滤未分配 ID 的条目（若存在）
        valid: Optional[np.ndarray] = None
        if tids.dtype == object:
            valid = np.array([x is not None for x in tids], dtype=bool)
        else:
            # 部分实现用 -1 表示无效
            valid = tids >= 0

        if not np.any(valid):
            return np.zeros((0, 5), dtype=np.float32)

        xyxy = xyxy[valid]
        tids = tids[valid]

        out = np.zeros((xyxy.shape[0], 5), dtype=np.float32)
        out[:, :4] = xyxy.astype(np.float32, copy=False)
        out[:, 4] = np.asarray(tids, dtype=np.float32).reshape(-1)

        for tid_f in out[:, 4]:
            tid = int(tid_f)
            if tid not in self.track_history:
                self.track_history[tid] = {"first_seen": now, "last_seen": now}
            else:
                self.track_history[tid]["last_seen"] = now

        return out
