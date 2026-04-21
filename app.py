"""
商场人流检测系统 —— Flask 主程序。

提供：MJPEG 视频流、当前统计 JSON、近 24 小时人流量历史（供 ECharts）。
启动时在后台线程运行 ``VideoEngine``（需 Redis、MySQL、OpenCV/GeCo2 环境可用）。

环境变量（可选）：
- ``SQLALCHEMY_DATABASE_URI``：覆盖默认 MySQL 连接串
- ``VIDEO_SOURCE``：视频源；未设置时默认使用下方测试用 MP4。也可设为 ``0`` 表示摄像头。
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta

import redis
from flask import Flask, Response, jsonify

from db_models import FlowRecord, init_db
from video_engine import VideoEngine

# ---------------------------------------------------------------------------
# 默认 MySQL（可通过环境变量覆盖）
# ---------------------------------------------------------------------------
DEFAULT_SQLALCHEMY_URI = (
    "mysql+pymysql://root:123456@localhost:3306/mall_cv?charset=utf8mb4"
)


# 默认测试视频（与 main_day1 / main_day2 一致）；改用摄像头时请设环境变量 VIDEO_SOURCE=0 或改默认
DEFAULT_TEST_VIDEO = r"E:\毕设视频\VID_20260417_114836.mp4"


def _parse_video_source() -> str | int:
    """从环境变量 VIDEO_SOURCE 解析；未设置时使用 DEFAULT_TEST_VIDEO。"""
    # 备用：默认摄像头（实时调试时可将下行改为 default_src = "0" 并注释掉 DEFAULT_TEST_VIDEO）
    # default_src = "0"
    default_src = DEFAULT_TEST_VIDEO
    raw = os.environ.get("VIDEO_SOURCE", default_src).strip()
    if raw.isdigit():
        return int(raw)
    return raw


def _redis_client() -> redis.Redis:
    """与 ``VideoEngine`` 一致的 Redis 连接（二进制，便于读取 JPEG）。"""
    return redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
        "SQLALCHEMY_DATABASE_URI", DEFAULT_SQLALCHEMY_URI
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    # 长连接场景下避免 MySQL 空闲断开
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True}

    init_db(app)

    rdb = _redis_client()

    @app.route("/api/video_feed")
    def api_video_feed():
        """
        Multipart MJPEG 推流：不断从 Redis ``latest_frame`` 读取 JPEG 字节并推送。
        前端可用 ``<img src="/api/video_feed">`` 或 fetch + blob 播放。
        """

        boundary = b"frame"

        def generate() -> bytes:
            while True:
                try:
                    jpeg_bytes = rdb.get("latest_frame")
                except redis.RedisError:
                    time.sleep(0.1)
                    continue
                if jpeg_bytes:
                    # multipart/x-mixed-replace 分片格式
                    yield (
                        b"--" + boundary + b"\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                    )
                else:
                    # 尚无画面时短暂等待，避免空转占满 CPU
                    time.sleep(0.05)

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={
                # 部分浏览器/代理对缓存敏感
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/api/stats/current")
    def api_stats_current():
        """读取 Redis ``current_stats``（JSON 字符串），原样解析后返回。"""
        try:
            raw = rdb.get("current_stats")
        except redis.RedisError as exc:
            return jsonify({"error": "redis_error", "detail": str(exc)}), 503
        if raw is None:
            return jsonify(
                {
                    "person_count": None,
                    "fps": None,
                    "message": "暂无统计数据（VideoEngine 可能未写入）",
                }
            )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return jsonify({"error": "invalid_json", "detail": str(exc)}), 500
        return jsonify(payload)

    @app.route("/api/stats/history")
    def api_stats_history():
        """
        查询 ``FlowRecord`` 最近 24 小时内记录，按时间正序，供 ECharts 折线图使用。

        每条包含：timestamp（ISO8601）、total_count、stay_count、region_name。
        """
        since = datetime.now() - timedelta(hours=24)
        rows = (
            FlowRecord.query.filter(FlowRecord.timestamp >= since)
            .order_by(FlowRecord.timestamp.asc())
            .all()
        )
        data = [
            {
                "id": row.id,
                "timestamp": row.timestamp.isoformat(),
                "region_name": row.region_name,
                "total_count": row.total_count,
                "stay_count": row.stay_count,
            }
            for row in rows
        ]
        return jsonify(data)

    return app


app = create_app()


if __name__ == "__main__":
    video_src = _parse_video_source()
    try:
        engine = VideoEngine(video_source=video_src, flask_app=app)
        engine.start()
        print(f"[app] VideoEngine 已启动（daemon=True），video_source={video_src!r}")
    except Exception as exc:
        print(f"[app] VideoEngine 启动失败（将仅提供无视频流的 API）：{exc}")

    # threaded=True 便于 MJPEG 长连接与其它 API 并发
    app.run(host="0.0.0.0", port=5000, threaded=True)
