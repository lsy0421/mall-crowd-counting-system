"""
Day1 主流程：本地视频 -> 预处理 -> GeCo2 检测 -> ByteTrack 跟踪 -> 可视化与保存。

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
OUTPUT_AVI = Path(__file__).resolve().parent / "output_day1.avi"


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

        writer = _open_writer(OUTPUT_AVI, fps, (w, h))

        detector = GeCo2Detector()
        tracker = TrackerManager(frame_rate=max(1, int(round(fps))))

        # Mock 范例框（与帧同尺寸的像素坐标 xyxy；可按实际画面调整）
        # ------------------ 替换原来的 mock_exemplars 部分 ------------------
        # 1. 先单独读取第一帧用于人工框选范例
        ret_first, first_frame = cap.read()
        if not ret_first:
            raise RuntimeError("无法读取视频的第一帧来设置范例")
            
        print("请在弹出的窗口中，用鼠标框选 2-3 个典型的行人。")
        print("操作说明：按住鼠标左键拖动框选 -> 按 'Space' 或 'Enter' 确认当前框 -> 选完所有目标后按 'Esc' 退出框选界面。")
        
        # 使用 OpenCV 的 selectROIs 让用户手动画框
        # 返回格式为 (x, y, w, h) 的列表
        rois = cv2.selectROIs("Select Exemplars (Press Esc to finish)", first_frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select Exemplars (Press Esc to finish)")
        
        if len(rois) == 0:
            raise ValueError("你没有框选任何目标，GeCo2 无法进行检测！")

        # 2. 将 (x, y, w, h) 转换为 GeCo2 需要的 [x1, y1, x2, y2]
        real_exemplars = []
        for rect in rois:
            x_min, y_min, box_w, box_h = rect
            real_exemplars.append([
                float(x_min), 
                float(y_min), 
                float(x_min + box_w), 
                float(y_min + box_h)
            ])
            
        detector.set_exemplars(real_exemplars)
        print(f"已动态设置真实 exemplars 坐标: {real_exemplars}")
        
        # 将视频指针重置回开头，以免漏掉第一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # ------------------------------------------------------------------
       

        window = "day1 GeCo2 + ByteTrack"
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
