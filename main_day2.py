"""
Day2 主流程：第一帧手动框选范例 -> 提取并锁死 GeCo2 范例特征 -> 全视频检测与 ByteTrack 跟踪。

依赖：OpenCV、PyTorch、supervision、GeCo2 环境；中文叠字需 Pillow（可选，缺失时回退英文）。
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from geco_detector import GeCo2Detector
from tracker_pipeline import TrackerManager, preprocess_frame

# 输入视频（本地测试）
# INPUT_VIDEO = 0  # 摄像头索引，备用
# INPUT_VIDEO = r"D:\videos\other.mp4"  # 其它素材，备用
INPUT_VIDEO = r"E:\毕设视频\VID_20260417_114836.mp4"
# 输出视频（与脚本同目录）
OUTPUT_AVI = Path(__file__).resolve().parent / "output_day2.avi"

SELECT_WINDOW = "Select Exemplars (Space=confirm, Esc=finish)"


def _draw_label_top_left(
    image: np.ndarray,
    text_cn: str,
    text_en: str,
    org: tuple[int, int] = (10, 36),
    color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """在图像左上角绘制文字；优先中文（PIL + 系统字体），失败则用英文 OpenCV 默认字体。"""
    try:
        from PIL import Image, ImageDraw, ImageFont

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        font = None
        for fp in (
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\msyhbd.ttc",
        ):
            try:
                font = ImageFont.truetype(fp, 26)
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()
        draw.text(org, text_cn, font=font, fill=(color[2], color[1], color[0]))
        image[:, :, :] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(
            image,
            text_en,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    w, h = size
    candidates = [
        ("XVID", cv2.VideoWriter_fourcc(*"XVID")),
        ("MJPG", cv2.VideoWriter_fourcc(*"MJPG")),
    ]
    for name, fourcc in candidates:
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if writer.isOpened():
            print(f"VideoWriter: fourcc={name}, path={path}")
            return writer
        writer.release()
    raise RuntimeError(f"无法创建 VideoWriter: {path}")


def _rois_to_xyxy(rois: np.ndarray) -> list[list[float]]:
    """将 selectROIs 返回的 (x, y, w, h) 转为 [x1, y1, x2, y2] 列表。"""
    real: list[list[float]] = []
    for rect in rois:
        x, y, bw, bh = [float(v) for v in rect]
        real.append([x, y, x + bw, y + bh])
    return real


def run() -> None:
    cap: cv2.VideoCapture | None = None
    writer: cv2.VideoWriter | None = None

    try:
        cap = cv2.VideoCapture(INPUT_VIDEO)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {INPUT_VIDEO}")

        fps_prop = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps_prop) if fps_prop and fps_prop > 1e-3 else 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            raise RuntimeError("无法读取视频宽高")

        ret_first, first_frame = cap.read()
        if not ret_first or first_frame is None:
            raise RuntimeError("无法读取视频的第一帧用于范例框选")

        print(
            "请在弹出的窗口中用鼠标框选 2～3 个行人范例：\n"
            "  拖动框选后按 Space（或 Enter）确认当前框，可继续框选多个；\n"
            "  全部选完后按 Esc 结束。"
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
            raise ValueError("未框选任何 ROI，GeCo2 无法设置范例。")

        n = len(rois)
        if n < 2 or n > 3:
            raise ValueError(f"请选择 2～3 个行人范例，当前为 {n} 个。")

        real_exemplars = _rois_to_xyxy(np.asarray(rois, dtype=np.float64))
        print(f"范例框 xyxy（像素）: {real_exemplars}")

        detector = GeCo2Detector()
        print("正在提取第一帧范例特征并锁死到检测器 …")
        feature_dict = detector.extract_features(first_frame, real_exemplars)
        detector.set_precomputed_features(feature_dict, real_exemplars)
        print("已调用 set_precomputed_features，范例特征已缓存。")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        writer = _open_writer(OUTPUT_AVI, fps, (w, h))
        tracker = TrackerManager(frame_rate=max(1, int(round(fps))))

        window = "day2 GeCo2 + ByteTrack (locked exemplar feats)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        frame_idx = 0
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            try:
                proc = preprocess_frame(frame)
                dets = detector.detect(proc)
                tracks = tracker.update(dets, frame)
            except Exception:
                print(f"[帧 {frame_idx}] 检测/跟踪失败:")
                traceback.print_exc()
                proc = frame
                dets = np.zeros((0, 6), dtype=np.float32)
                tracks = np.zeros((0, 5), dtype=np.float32)

            vis = frame.copy()
            n_person = tracks.shape[0]
            for row in tracks:
                x1, y1, x2, y2, tid_f = row
                tid = int(tid_f)
                p1 = (int(round(x1)), int(round(y1)))
                p2 = (int(round(x2)), int(round(y2)))
                cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
                label = f"ID: {tid}"
                cv2.putText(
                    vis,
                    label,
                    (p1[0], max(0, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            _draw_label_top_left(
                vis,
                f"当前画面人数：{n_person}",
                f"People: {n_person}",
            )

            writer.write(vis)
            cv2.imshow(window, vis)

            elapsed = time.perf_counter() - t0
            inst_fps = 1.0 / elapsed if elapsed > 1e-9 else 0.0
            print(
                f"帧 {frame_idx}: 耗时 {elapsed * 1000:.2f} ms, "
                f"瞬时 FPS: {inst_fps:.2f}, 人数: {n_person}"
            )

            frame_idx += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                print("用户按 q 退出。")
                break

    except KeyboardInterrupt:
        print("中断退出。")
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("VideoCapture / VideoWriter 已释放，窗口已关闭。")


if __name__ == "__main__":
    run()
