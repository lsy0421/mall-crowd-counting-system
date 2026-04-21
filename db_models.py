"""
商场人流检测系统 —— 数据库模型定义（Flask-SQLAlchemy + MySQL）。

本文件命名为 ``db_models.py`` 而非 ``models.py``，避免与 GeCo2 官方包内的
``models`` 目录冲突（否则 ``import models`` 会指向本文件，导致
``models.counter_infer`` 导入失败）。

使用说明（简要）：
1. 在 Flask 应用配置中设置 ``SQLALCHEMY_DATABASE_URI``，例如 ::
       mysql+pymysql://用户名:密码@主机:3306/数据库名?charset=utf8mb4
2. 在应用工厂或主模块中调用 ``db.init_app(app)`` 与 ``init_db(app)`` 完成初始化与建表。

依赖：``Flask-SQLAlchemy``，MySQL 驱动常用 ``PyMySQL``（与上述 URI  scheme 对应）。
"""

from __future__ import annotations

import os

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash

# ---------------------------------------------------------------------------
# 全局 SQLAlchemy 扩展实例
# ---------------------------------------------------------------------------
# 不在此处绑定 app，而是在应用创建后通过 ``db.init_app(app)`` 绑定，
# 便于测试与多应用实例场景。
db = SQLAlchemy()


class User(db.Model):
    """
    管理员账号表。

    存储后台登录用的用户名与密码哈希（永远不要明文保存密码）。
    """

    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True, comment="主键")
    username = db.Column(
        db.String(64),
        unique=True,
        nullable=False,
        index=True,
        comment="登录用户名，全局唯一",
    )
    password_hash = db.Column(db.String(255), nullable=False, comment="密码哈希（werkzeug 生成）")

    def __repr__(self) -> str:
        return f"<User {self.id} {self.username!r}>"


class FlowRecord(db.Model):
    """
    人流量历史记录表。

    ``timestamp`` 表示统计时段所对应的时间点，业务上约定精度为「分钟」
    （同一分钟内同一区域可聚合或覆盖，由业务层决定）；数据库存储为 ``DateTime``，
    写入时请将秒（及微秒）置为 0，以保证「精确到分」的语义一致。
    """

    __tablename__ = "flow_records"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True, comment="主键")
    # 精确到分钟：建议在业务代码中使用 datetime.replace(second=0, microsecond=0)
    timestamp = db.Column(
        db.DateTime,
        nullable=False,
        index=True,
        comment="统计时段时间戳（精度：分钟）",
    )
    region_name = db.Column(
        db.String(128),
        nullable=False,
        default="Main",
        comment="摄像头/区域名称，默认 Main",
    )
    total_count = db.Column(db.Integer, nullable=False, default=0, comment="该时段检测到的人数总计")
    stay_count = db.Column(db.Integer, nullable=False, default=0, comment="该时段滞留人数")

    def __repr__(self) -> str:
        return f"<FlowRecord {self.timestamp} {self.region_name} total={self.total_count}>"


def init_db(app) -> None:
    """
    初始化数据库：将本模块的 ``db`` 绑定到 Flask ``app``，创建所有表，
    并在不存在默认管理员时插入一条 admin 记录。

    :param app: 已配置 ``SQLALCHEMY_DATABASE_URI`` 的 Flask 应用实例。

    默认管理员账号：
    - 用户名：从环境变量 ``MALLSYS_ADMIN_USERNAME`` 读取，缺省为 ``admin``。
    - 密码：从环境变量 ``MALLSYS_ADMIN_PASSWORD`` 读取，缺省为 ``admin123``。
      **生产环境务必通过环境变量修改，并首次登录后改密。**
    """
    # 将 db 与当前应用绑定（可重复调用，但通常只调一次）
    db.init_app(app)

    # 必须在应用上下文中才能访问数据库
    with app.app_context():
        # 根据上述模型在数据库中创建缺失的表（已存在的表不会删除数据）
        db.create_all()

        admin_username = os.environ.get("MALLSYS_ADMIN_USERNAME", "admin")
        admin_password = os.environ.get("MALLSYS_ADMIN_PASSWORD", "admin123")

        # 若该用户名尚不存在，则插入默认管理员
        existing = User.query.filter_by(username=admin_username).first()
        if existing is None:
            default_user = User(
                username=admin_username,
                password_hash=generate_password_hash(admin_password),
            )
            db.session.add(default_user)
            db.session.commit()
            # 注意：生产环境不要在日志中打印明文密码
            print(
                f"[init_db] 已创建默认管理员账号: username={admin_username!r} "
                "(密码来自环境变量 MALLSYS_ADMIN_PASSWORD 或使用默认值，请尽快修改)"
            )
        else:
            print(f"[init_db] 管理员 {admin_username!r} 已存在，跳过插入。")
