# src/recall/itemcf.py
# 基于物品的协同过滤 —— 核心召回算法
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.sparse import csr_matrix

class ItemCF:
    """
    基于物品的协同过滤
    输入: 用户-歌曲播放记录（隐式反馈，play_count作为权重）
    输出: 歌曲相似度矩阵 → 给定用户的TopN召回列表
    """
    def __init__(self, n_similar: int = 20, alpha: float = 0.5):
        self.n_similar = n_similar  # 每首歌保留的最相似歌曲数
        self.alpha = alpha          # 热门物品惩罚系数

    def fit(self, df: pd.DataFrame):
        """
        df 需包含列: user_id, song_id, play_count
        """
        # 对数平滑播放次数（隐式反馈处理）
        df = df.copy()
        df["weight"] = np.log1p(df["play_count"])

        self.user_songs = df.groupby("user_id").apply(
            lambda x: dict(zip(x["song_id"], x["weight"]))
        ).to_dict()

        # 构建倒排索引: song_id → {user_id: weight}
        song_users = defaultdict(dict)
        for _, row in df.iterrows():
            song_users[row["song_id"]][row["user_id"]] = row["weight"]

        # 计算歌曲共现矩阵（带热门惩罚）
        co_occur  = defaultdict(lambda: defaultdict(float))
        song_cnt  = {s: len(users) for s, users in song_users.items()}

        for song_i, users_i in song_users.items():
            for uid, w_i in users_i.items():
                for song_j, w_j in self.user_songs.get(uid, {}).items():
                    if song_i == song_j:
                        continue
                    co_occur[song_i][song_j] += w_i * w_j

        # 归一化 → 相似度（含热门惩罚）
        self.sim = {}
        for song_i, related in co_occur.items():
            sims = {}
            for song_j, cnt in related.items():
                denom = (song_cnt[song_i] ** self.alpha *
                         song_cnt[song_j] ** (1 - self.alpha))
                sims[song_j] = cnt / (denom + 1e-9)
            # 只保留TopN相似歌曲
            self.sim[song_i] = sorted(sims.items(),
                                       key=lambda x: -x[1])[:self.n_similar]
        return self

    def recommend(self, user_id: str, topk: int = 50) -> list:
        """返回用户的TopK召回歌曲列表"""
        played = self.user_songs.get(user_id, {})
        scores = defaultdict(float)
        for song_i, w in played.items():
            for song_j, sim_score in self.sim.get(song_i, []):
                if song_j not in played:  # 过滤已听过的歌
                    scores[song_j] += sim_score * w
        return sorted(scores.items(), key=lambda x: -x[1])[:topk]


# ─────────────────────────────────────────────────────────────
# src/recall/matrix_factorization.py
# ALS（交替最小二乘）矩阵分解 —— 适合隐式反馈大规模场景
# ─────────────────────────────────────────────────────────────
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

class ALSRecall:
    """
    使用Spark MLlib的ALS算法
    MSD的Taste Profile有48M行，适合分布式ALS
    """
    def __init__(self, rank: int = 64, max_iter: int = 10,
                 reg_param: float = 0.1, alpha: float = 40.0):
        self.rank       = rank        # 隐向量维度
        self.max_iter   = max_iter
        self.reg_param  = reg_param
        self.alpha      = alpha       # 隐式反馈置信度系数

    def fit(self, spark: SparkSession, df_plays_path: str):
        df = spark.read.parquet(df_plays_path)
        # ALS需要整数ID，做映射
        from pyspark.ml.feature import StringIndexer
        user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx")
        song_indexer = StringIndexer(inputCol="song_id", outputCol="song_idx")
        df = user_indexer.fit(df).transform(df)
        df = song_indexer.fit(df).transform(df)

        als = ALS(
            rank=self.rank,
            maxIter=self.max_iter,
            regParam=self.reg_param,
            implicitPrefs=True,    # 隐式反馈模式
            alpha=self.alpha,
            userCol="user_idx",
            itemCol="song_idx",
            ratingCol="play_count",
            coldStartStrategy="drop"
        )
        self.model = als.fit(df)
        return self

    def recommend_for_all_users(self, topk: int = 50):
        """批量为所有用户生成召回列表，结果存入在线存储"""
        return self.model.recommendForAllUsers(topk)

    def get_embeddings(self):
        """提取用户/歌曲Embedding，用于近似最近邻检索（Faiss）"""
        user_emb = self.model.userFactors  # DataFrame: id, features
        song_emb = self.model.itemFactors
        return user_emb, song_emb


# ─────────────────────────────────────────────────────────────
# src/recall/content_based.py
# 基于内容的召回 —— 利用MSD音频特征（冷启动新歌场景）
# ─────────────────────────────────────────────────────────────
import faiss
import numpy as np

class ContentBasedRecall:
    """
    利用MSD的音频特征（timbre, tempo, loudness等）
    构建歌曲向量索引，通过近似最近邻召回相似歌曲
    用途: 新歌冷启动 / 基于当前播放歌曲的实时推荐
    """
    def __init__(self, n_list: int = 100):
        self.n_list = n_list  # Faiss IVF聚类中心数

    def build_index(self, song_ids: list, feature_matrix: np.ndarray):
        """
        song_ids: List[str]
        feature_matrix: shape (n_songs, feature_dim)
        """
        self.song_ids = song_ids
        self.id2idx   = {sid: i for i, sid in enumerate(song_ids)}
        dim = feature_matrix.shape[1]
        # 归一化（用余弦相似度）
        faiss.normalize_L2(feature_matrix)
        # 构建IVF索引（大规模近似检索）
        quantizer  = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, self.n_list,
                                         faiss.METRIC_INNER_PRODUCT)
        self.index.train(feature_matrix)
        self.index.add(feature_matrix)
        self.feature_matrix = feature_matrix

    def recall_by_song(self, song_id: str, topk: int = 50) -> list:
        """给定一首歌，找最相似的topk首"""
        idx   = self.id2idx.get(song_id)
        if idx is None:
            return []
        query = self.feature_matrix[idx:idx+1].copy()
        faiss.normalize_L2(query)
        D, I = self.index.search(query, topk + 1)
        results = [(self.song_ids[i], float(D[0][j]))
                   for j, i in enumerate(I[0]) if i != idx]
        return results[:topk]
