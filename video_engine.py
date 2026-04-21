"""
后台视频分析引擎：在独立线程中运行 GeCo2 + ByteTrack，将画面与统计推送到 Redis，
并每分钟将平均人数写入 MySQL（FlowRecord）。

使用方式（示例）::

    app = create_app()
    with app.app_context():
        init_db(app)
    engine = VideoEngine(video_source=0, flask_app=app)
    engine.start()

注意：``cv2.selectROIs`` 依赖 OpenCV 高频窗口，在部分系统上需在主线程调用；
若框选窗口异常，可改为预先传入范例框或使用视频文件首帧在调试阶段完成框选。
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Optional

import cv2
import numpy as np
import redis

from geco_detector import GeCo2Detector
from tracker_pipeline import TrackerManager, preprocess_frame

# MySQL / SQLAlchemy（需在 Flask 应用中已执行 db.init_app）
from db_models import FlowRecord, db

SELECT_WINDOW = "VideoEngine — Select Exemplars (Space / Enter, Esc finish)"


class VideoEngine(threading.Thread):
    """
    在后台线程中持续读取视频、检测、跟踪，并向 Redis 与数据库分发结果。
    """

    def __init__(
        self,
        video_source: str | int,
        flask_app: Any | None = None,
        *,
        region_name: str = "Main",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        stats_interval_sec: float = 60.0,
        jpeg_quality: int = 85,
    ) -> None:
        """
        :param video_source: 视频路径字符串，或摄像头索引 ``0``、``1`` …
        :param flask_app: Flask 应用实例；若提供则每分钟写入 ``FlowRecord``（需已配置数据库 URI 且 ``db.init_app``）。
        :param region_name: 写入 ``FlowRecord.region_name`` 的区域名。
        :param redis_host / redis_port / redis_db: Redis 连接参数。
        :param stats_interval_sec: 聚合写库周期（秒），默认 60。
        :param jpeg_quality: ``imencode`` JPEG 质量 0–100。
        """
        super().__init__(daemon=True)
        self._video_source = video_source
        self._flask_app = flask_app
        self._region_name = region_name
        self._stats_interval_sec = float(stats_interval_sec)
        self._jpeg_quality = int(jpeg_quality)

        # Redis 客户端（失败时允许后续重试或仅打日志）
        try:
            self._redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=False,
            )
            self._redis.ping()
        except redis.RedisError as exc:
            raise RuntimeError(f"无法连接 Redis ({redis_host}:{redis_port}/{redis_db}): {exc}") from exc

        self._cap = cv2.VideoCapture(self._video_source)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self._video_source!r}")

        fps_prop = self._cap.get(cv2.CAP_PROP_FPS)
        self._video_fps = float(fps_prop) if fps_prop and fps_prop > 1e-3 else 30.0

        self._detector = GeCo2Detector()
        self._tracker = TrackerManager(frame_rate=max(1, int(round(self._video_fps))))

        self._stop_event = threading.Event()
        self._frame_index = 0

        # 每分钟写库：缓存该时间窗口内每一帧的人数，用于算术平均
        self._count_buffer: list[int] = []
        self._last_flush_monotonic: float | None = None

    def stop(self) -> None:
        """请求线程结束循环（下一帧检测后退出）。"""
        self._stop_event.set()

    @staticmethod
    def _rois_to_xyxy(rois: np.ndarray) -> list[list[float]]:
        out: list[list[float]] = []
        for rect in rois:
            x, y, bw, bh = (float(v) for v in rect)
            out.append([x, y, x + bw, y + bh])
        return out

    def _draw_vis(self, frame: np.ndarray, tracks: np.ndarray, n_person: int) -> np.ndarray:
        """在画面上绘制跟踪框与人数（不依赖 Pillow，便于后台环境）。"""
        vis = frame.copy()
        for row in tracks:
            x1, y1, x2, y2, tid_f = row
            tid = int(tid_f)
            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))
            cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"ID: {tid}",
                (p1[0], max(0, p1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            vis,
            f"People: {n_person}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return vis

    def _push_redis_frame(self, vis_frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(
            ".jpg",
            vis_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality],
        )
        if not ok:
            print("[VideoEngine] cv2.imencode 失败，跳过写入 latest_frame")
            return
        try:
            # 二进制 JPEG，便于前端直接作为 image/jpeg 返回
            self._redis.set("latest_frame", buf.tobytes())
        except redis.RedisError:
            print("[VideoEngine] Redis 写入 latest_frame 失败:")
            traceback.print_exc()

    def _push_redis_stats(self, payload: dict[str, Any]) -> None:
        try:
            self._redis.set("current_stats", json.dumps(payload, ensure_ascii=False))
        except (TypeError, redis.RedisError):
            print("[VideoEngine] Redis 写入 current_stats 失败:")
            traceback.print_exc()

    def _maybe_flush_flow_record(self) -> None:
        """
        若距离上次写库已超过 ``stats_interval_sec``，则计算窗口内人数平均值并写入 MySQL。
        """
        if self._flask_app is None:
            return
        now = time.monotonic()
        if self._last_flush_monotonic is None:
            self._last_flush_monotonic = now
            return
        if now - self._last_flush_monotonic < self._stats_interval_sec:
            return

        if self._count_buffer:
            avg_count = sum(self._count_buffer) / len(self._count_buffer)
        else:
            avg_count = 0.0

        total_int = int(round(avg_count))
        # 滞留人数业务规则未接入时记 0；若需与峰值/规则关联可在此替换
        stay_int = 0

        try:
            with self._flask_app.app_context():
                minute_ts = datetime.now().replace(second=0, microsecond=0)
                row = FlowRecord(
                    timestamp=minute_ts,
                    region_name=self._region_name,
                    total_count=total_int,
                    stay_count=stay_int,
                )
                db.session.add(row)
                db.session.commit()
                print(
                    f"[VideoEngine] FlowRecord 已写入: ts={minute_ts}, "
                    f"region={self._region_name!r}, avg_person≈{avg_count:.2f} -> total_count={total_int}"
                )
        except Exception:
            print("[VideoEngine] 写入 FlowRecord 失败:")
            traceback.print_exc()
            try:
                with self._flask_app.app_context():
                    db.session.rollback()
            except Exception:
                traceback.print_exc()
        finally:
            self._count_buffer.clear()
            self._last_flush_monotonic = now

    def run(self) -> None:
        cap = self._cap
        try:
            ret0, first_frame = cap.read()
            if not ret0 or first_frame is None:
                print("[VideoEngine] 无法读取第一帧，线程退出。")
                return

            print(
                "[VideoEngine] 请在窗口中框选 2～3 个行人范例："
                "拖动选框后 Space/Enter 确认，Esc 结束。"
            )
            rois = cv2.selectROIs(
                SELECT_WINDOW,
                first_frame,
                showCrosshair=True,
                fromCenter=False,
            )
            try:
                cv2.destroyWindow(SELECT_WINDOW)
            except cv2.error:
                pass

            if rois is None or len(rois) == 0:
                print("[VideoEngine] 未选择任何 ROI，线程退出。")
                return
            n_roi = len(rois)
            if n_roi < 2 or n_roi > 3:
                print(f"[VideoEngine] 需要 2～3 个范例，当前 {n_roi} 个，线程退出。")
                return

            real_exemplars = self._rois_to_xyxy(np.asarray(rois, dtype=np.float64))
            print(f"[VideoEngine] 范例 xyxy: {real_exemplars}")

            try:
                feat = self._detector.extract_features(first_frame, real_exemplars)
                self._detector.set_precomputed_features(feat, real_exemplars)
            except Exception:
                print("[VideoEngine] 提取/锁死范例特征失败:")
                traceback.print_exc()
                return

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._last_flush_monotonic = time.monotonic()

            while not self._stop_event.is_set():
                t_loop = time.perf_counter()
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[VideoEngine] 视频结束或读取失败，退出循环。")
                    break

                try:
                    proc = preprocess_frame(frame)
                    dets = self._detector.detect(proc)
                    tracks = self._tracker.update(dets, frame)
                except Exception:
                    print(f"[VideoEngine] 帧 {self._frame_index} 检测/跟踪异常:")
                    traceback.print_exc()
                    tracks = np.zeros((0, 5), dtype=np.float32)

                n_person = int(tracks.shape[0])
                self._count_buffer.append(n_person)

                vis_frame = self._draw_vis(frame, tracks, n_person)

                elapsed = time.perf_counter() - t_loop
                inst_fps = 1.0 / elapsed if elapsed > 1e-9 else 0.0

                self._push_redis_frame(vis_frame)
                self._push_redis_stats(
                    {
                        "person_count": n_person,
                        "fps": round(inst_fps, 3),
                        "frame_index": self._frame_index,
                        "timestamp": time.time(),
                        "region": self._region_name,
                        "video_fps": round(self._video_fps, 3),
                    }
                )

                self._maybe_flush_flow_record()

                self._frame_index += 1

        except Exception:
            print("[VideoEngine] run() 未捕获异常:")
            traceback.print_exc()
        finally:
            try:
                cap.release()
            except Exception:
                pass
            print("[VideoEngine] 已释放 VideoCapture，线程结束。")
