"""
模型版本管理器
===============
- 每次训练自动创建带时间戳的版本目录，旧模型不再被覆盖
- 维护 registry.json 注册表，记录所有版本及元数据
- 支持切换活跃版本、删除旧版本
- 向后兼容：活跃版本的文件始终同步到 saved/ 根目录，predictor 无需改动即可工作

目录结构:
  model/saved/
  ├── registry.json            # 版本注册表
  ├── outcome_model.json       # 活跃版本文件（根目录副本，向后兼容）
  ├── home_score_model.json
  ├── ...
  └── versions/
      ├── v_20260618_160000/
      │   ├── outcome_model.json
      │   ├── home_score_model.json
      │   ├── away_score_model.json
      │   ├── feature_columns.json
      │   ├── elo_ratings.json
      │   └── training_metadata.json
      └── v_20260618_170000/
          └── ...
"""
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional


# 模型产物文件清单（每次训练都会产生的固定文件集）
MODEL_FILES = [
    'outcome_model.json',
    'home_score_model.json',
    'away_score_model.json',
    'feature_columns.json',
    'elo_ratings.json',
    'training_metadata.json',
]


class ModelVersionManager:
    """模型版本管理器"""

    def __init__(self, model_dir: Path):
        """
        Args:
            model_dir: model/saved/ 目录的 Path
        """
        self.model_dir = Path(model_dir)
        self.versions_dir = self.model_dir / 'versions'
        self.registry_path = self.model_dir / 'registry.json'
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 注册表读写
    # ------------------------------------------------------------------

    def _load_registry(self) -> dict:
        """加载注册表，不存在则返回空结构"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"active_version": None, "versions": []}

    def _save_registry(self, registry: dict):
        """保存注册表"""
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 版本创建 & 激活
    # ------------------------------------------------------------------

    def create_version(self, metadata: dict, note: str = "") -> tuple[str, Path]:
        """
        创建一个新版本目录（不写入文件，仅建目录 + 注册）。
        调用方负责把模型文件写入返回的 version_dir。
        创建后自动设为活跃版本。

        Args:
            metadata: 训练元数据（accuracy、train_size 等）
            note: 可选的用户备注

        Returns:
            (version_id, version_dir) — 版本ID 和版本目录路径
        """
        version_id = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        version_dir = self.versions_dir / version_id

        # 防止同一秒内重复创建（极端情况）
        counter = 1
        while version_dir.exists():
            version_id = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{counter}"
            version_dir = self.versions_dir / version_id
            counter += 1

        version_dir.mkdir(parents=True, exist_ok=True)

        # 注册
        registry = self._load_registry()
        version_entry = {
            "version_id": version_id,
            "created_at": datetime.now().isoformat(),
            "note": note,
            "metadata": metadata,
        }
        registry["versions"].append(version_entry)
        registry["active_version"] = version_id
        self._save_registry(registry)

        return version_id, version_dir

    def finalize_version(self, version_id: str):
        """
        确认版本文件已写入完毕，将其同步到根目录（设为活跃版本）。
        通常在 train.py 把所有模型文件写入 version_dir 后调用。
        """
        version_dir = self.versions_dir / version_id
        if not version_dir.exists():
            raise FileNotFoundError(f"版本目录不存在: {version_dir}")

        # 同步文件到根目录（向后兼容 predictor 的直接路径加载）
        self._sync_to_root(version_id)

        # 确保注册表中标记为活跃
        registry = self._load_registry()
        registry["active_version"] = version_id
        self._save_registry(registry)

        print(f"[版本管理] 版本 {version_id} 已设为活跃，文件已同步到根目录")

    def _sync_to_root(self, version_id: str):
        """将指定版本的文件复制到根目录（覆盖现有文件）"""
        version_dir = self.versions_dir / version_id
        for fname in MODEL_FILES:
            src = version_dir / fname
            if src.exists():
                shutil.copy2(src, self.model_dir / fname)

    def activate_version(self, version_id: str) -> bool:
        """
        切换活跃版本：把指定版本的文件同步到根目录。
        Returns: True 成功, False 版本不存在
        """
        version_dir = self.versions_dir / version_id
        if not version_dir.exists():
            return False

        # 验证版本文件完整性
        missing = [f for f in MODEL_FILES if not (version_dir / f).exists()]
        if missing:
            raise ValueError(f"版本 {version_id} 缺少文件: {missing}")

        self._sync_to_root(version_id)

        registry = self._load_registry()
        registry["active_version"] = version_id
        self._save_registry(registry)

        print(f"[版本管理] 已切换活跃版本到 {version_id}")
        return True

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_active_version_id(self) -> Optional[str]:
        """获取当前活跃版本ID"""
        registry = self._load_registry()
        return registry.get("active_version")

    def get_active_version_dir(self) -> Optional[Path]:
        """获取当前活跃版本的目录路径"""
        vid = self.get_active_version_id()
        if vid is None:
            return None
        d = self.versions_dir / vid
        return d if d.exists() else None

    def list_versions(self) -> list[dict]:
        """
        列出所有版本（按创建时间降序），附带 active 标记。
        """
        registry = self._load_registry()
        active = registry.get("active_version")
        versions = registry.get("versions", [])
        # 降序排列（最新的在前）
        versions = sorted(versions, key=lambda v: v.get("created_at", ""), reverse=True)
        for v in versions:
            v["active"] = (v["version_id"] == active)
        return versions

    def get_version_info(self, version_id: str) -> Optional[dict]:
        """获取单个版本的详细信息"""
        registry = self._load_registry()
        for v in registry.get("versions", []):
            if v["version_id"] == version_id:
                v["active"] = (v["version_id"] == registry.get("active_version"))
                return v
        return None

    # ------------------------------------------------------------------
    # 删除
    # ------------------------------------------------------------------

    def delete_version(self, version_id: str) -> bool:
        """
        删除一个版本（不能删除活跃版本）。
        Returns: True 成功, False 版本不存在或是活跃版本
        """
        registry = self._load_registry()
        if registry.get("active_version") == version_id:
            raise ValueError(f"不能删除活跃版本 {version_id}，请先切换到其他版本")

        # 从注册表移除
        original_len = len(registry["versions"])
        registry["versions"] = [
            v for v in registry["versions"] if v["version_id"] != version_id
        ]
        if len(registry["versions"]) == original_len:
            return False  # 版本不存在

        self._save_registry(registry)

        # 删除版本目录
        version_dir = self.versions_dir / version_id
        if version_dir.exists():
            shutil.rmtree(version_dir)

        print(f"[版本管理] 已删除版本 {version_id}")
        return True

    # ------------------------------------------------------------------
    # 迁移现有模型
    # ------------------------------------------------------------------

    def adopt_existing_models(self, note: str = "迁移现有模型为初始版本") -> Optional[str]:
        """
        将根目录下已有的模型文件注册为一个版本（用于首次启用版本管理时）。
        如果注册表已有版本则跳过。
        Returns: version_id 或 None（如果已有版本或缺少文件）
        """
        registry = self._load_registry()
        if registry["versions"]:
            print("[版本管理] 注册表已有版本，跳过迁移")
            return None

        # 检查根目录文件是否齐全
        missing = [f for f in MODEL_FILES if not (self.model_dir / f).exists()]
        if missing:
            print(f"[版本管理] 根目录缺少文件 {missing}，无法迁移")
            return None

        # 读取现有元数据
        metadata_path = self.model_dir / 'training_metadata.json'
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # 创建版本
        version_id, version_dir = self.create_version(metadata, note)

        # 复制文件到版本目录
        for fname in MODEL_FILES:
            shutil.copy2(self.model_dir / fname, version_dir / fname)

        self.finalize_version(version_id)
        print(f"[版本管理] 现有模型已迁移为版本 {version_id}")
        return version_id
