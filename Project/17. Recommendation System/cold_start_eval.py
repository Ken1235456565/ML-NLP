# src/cold_start/new_user.py
# 新用户冷启动策略
import pandas as pd
import numpy as np

class NewUserColdStart:
    """
    MSD场景下新用户冷启动：
    无历史行为 → 利用注册信息/选歌页 → 粗粒度个性化推荐
    """
    def __init__(self, hot_songs_by_genre: dict,
                       content_recall=None):
        # 每个流派的热门歌曲列表（离线预计算）
        self.hot_by_genre   = hot_songs_by_genre
        self.content_recall = content_recall

    def recommend_by_genre_selection(self,
                                      selected_genres: list,
                                      topk: int = 30) -> list:
        """
        用户在引导页主动选择流派后的即时推荐
        selected_genres: e.g. ["pop", "rock", "jazz"]
        """
        result = []
        per_genre = max(1, topk // len(selected_genres))
        for genre in selected_genres:
            songs = self.hot_by_genre.get(genre, [])[:per_genre]
            result.extend(songs)
        return result[:topk]

    def recommend_by_seed_song(self, seed_song_id: str,
                                topk: int = 30) -> list:
        """
        用户播放第一首歌后，立即基于内容相似度扩展
        适用: 新用户从分享链接/搜索进入直接播放的场景
        """
        if self.content_recall:
            return [s for s, _ in
                    self.content_recall.recall_by_song(seed_song_id, topk)]
        return []


# src/cold_start/new_song.py
# 新歌冷启动策略
class NewSongColdStart:
    """
    新歌上架时无播放数据 → 依赖内容特征进入召回池
    """
    def __init__(self, content_recall, genre_hot_users: dict):
        self.content_recall  = content_recall
        self.genre_hot_users = genre_hot_users  # 流派→活跃用户列表

    def find_target_users(self, song_id: str,
                           song_genre: str,
                           topk_users: int = 1000) -> list:
        """
        新歌冷启动：找到最可能喜欢这首歌的用户群
        1. 先找内容相似的已有歌曲
        2. 再找这些歌曲的历史播放用户
        """
        similar_songs = [s for s, _ in
                         self.content_recall.recall_by_song(song_id, topk=20)]
        # 同时利用流派匹配活跃用户
        genre_users   = self.genre_hot_users.get(song_genre, [])[:topk_users]
        return {"similar_songs": similar_songs, "target_users": genre_users}


# ─────────────────────────────────────────────────────────────
# src/evaluation/offline_metrics.py
# 离线评估指标
# ─────────────────────────────────────────────────────────────
import numpy as np
from typing import Dict, List

def precision_at_k(recommended: list, ground_truth: set, k: int) -> float:
    rec_k = recommended[:k]
    hits  = sum(1 for s in rec_k if s in ground_truth)
    return hits / k

def recall_at_k(recommended: list, ground_truth: set, k: int) -> float:
    rec_k = recommended[:k]
    hits  = sum(1 for s in rec_k if s in ground_truth)
    return hits / len(ground_truth) if ground_truth else 0.0

def ndcg_at_k(recommended: list, ground_truth: set, k: int) -> float:
    """归一化折扣累计增益 —— 考虑排序位置的质量指标"""
    dcg = sum(1.0 / np.log2(i + 2)
              for i, s in enumerate(recommended[:k]) if s in ground_truth)
    idcg = sum(1.0 / np.log2(i + 2)
               for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0

def coverage(all_recommendations: list, all_songs: set) -> float:
    """覆盖度：被推荐到的歌曲占全库比例"""
    rec_songs = set(s for recs in all_recommendations for s in recs)
    return len(rec_songs) / len(all_songs)

def evaluate_model(model_recommend_fn,
                   test_users: Dict[str, set],
                   k_list: List[int] = [10, 20, 50]) -> dict:
    """
    批量评估推荐模型
    test_users: {user_id: set of held-out songs}
    """
    metrics = {f"P@{k}": [] for k in k_list}
    metrics.update({f"R@{k}": [] for k in k_list})
    metrics.update({f"NDCG@{k}": [] for k in k_list})

    for uid, truth in test_users.items():
        recs = [s for s, _ in model_recommend_fn(uid, topk=max(k_list))]
        for k in k_list:
            metrics[f"P@{k}"].append(precision_at_k(recs, truth, k))
            metrics[f"R@{k}"].append(recall_at_k(recs, truth, k))
            metrics[f"NDCG@{k}"].append(ndcg_at_k(recs, truth, k))

    return {key: np.mean(vals) for key, vals in metrics.items()}


# ─────────────────────────────────────────────────────────────
# src/evaluation/ab_test.py
# A/B测试流量切分
# ─────────────────────────────────────────────────────────────
import hashlib

class ABTestRouter:
    """
    基于用户ID哈希的确定性流量切分
    保证同一用户每次进入同一分组（正交性）
    """
    def __init__(self, experiments: dict):
        """
        experiments = {
            "exp_als_v2": {"traffic": 0.1, "model": "als_v2"},
            "exp_fm":     {"traffic": 0.1, "model": "fm"},
            "control":    {"traffic": 0.8, "model": "itemcf_base"},
        }
        """
        self.experiments = experiments
        # 归一化流量
        assert abs(sum(e["traffic"] for e in experiments.values()) - 1.0) < 1e-6

    def get_experiment(self, user_id: str, layer: str = "recall") -> str:
        """
        layer: 不同层（recall/rank/rerank）流量正交
        同一用户在不同层可进入不同实验
        """
        hash_key = f"{layer}:{user_id}"
        h = int(hashlib.md5(hash_key.encode()).hexdigest(), 16)
        bucket = (h % 1000) / 1000.0  # [0, 1)

        cumulative = 0.0
        for exp_name, cfg in self.experiments.items():
            cumulative += cfg["traffic"]
            if bucket < cumulative:
                return exp_name, cfg["model"]
        return "control", self.experiments["control"]["model"]
