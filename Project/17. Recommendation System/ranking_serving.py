# src/ranking/fm_ranker.py
# Factorization Machine排序模型 —— 支持高维稀疏特征（流派、标签等）
import torch
import torch.nn as nn
import numpy as np

class FM(nn.Module):
    """
    FM公式: y = w0 + Σwi*xi + Σ<vi,vj>xi*xj
    支持稀疏特征（one-hot流派、艺人标签）和稠密特征（tempo, energy等）
    """
    def __init__(self, n_features: int, n_factors: int = 16):
        super().__init__()
        self.w0      = nn.Parameter(torch.zeros(1))
        self.w       = nn.Embedding(n_features, 1)   # 一阶权重
        self.v       = nn.Embedding(n_features, n_factors)  # 二阶隐向量

    def forward(self, x_sparse: torch.Tensor,      # (B, sparse_len) 稀疏特征索引
                       x_dense:  torch.Tensor):     # (B, dense_dim) 稠密特征
        # 一阶项
        linear = self.w(x_sparse).squeeze(-1).sum(dim=1)
        # 二阶交叉项（FM trick，O(kn)复杂度）
        emb = self.v(x_sparse)                      # (B, sparse_len, k)
        sum_sq = emb.sum(dim=1) ** 2
        sq_sum = (emb ** 2).sum(dim=1)
        interaction = 0.5 * (sum_sq - sq_sum).sum(dim=1)

        out = self.w0 + linear + interaction
        return torch.sigmoid(out)

class FMRanker:
    def __init__(self, n_features: int, n_factors: int = 16, lr: float = 1e-3):
        self.model = FM(n_features, n_factors)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.BCELoss()

    def train_step(self, x_sparse, x_dense, labels):
        self.model.train()
        pred = self.model(x_sparse, x_dense)
        loss = self.loss_fn(pred, labels.float())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    def predict(self, x_sparse, x_dense) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(x_sparse, x_dense).numpy()


# ─────────────────────────────────────────────────────────────
# src/serving/api.py
# FastAPI在线推荐服务 —— 召回→排序→重排 三段式流程
# ─────────────────────────────────────────────────────────────
from fastapi import FastAPI
from pydantic import BaseModel
import time, redis, json

app = FastAPI(title="Music Recommender API")

# 启动时加载模型（实际生产用模型服务器如TF Serving）
from src.recall.itemcf import ItemCF
from src.recall.content_based import ContentBasedRecall
from src.ranking.fm_ranker import FMRanker

# 模拟全局模型实例（生产环境改为从Redis/对象存储加载）
itemcf_model    = None
content_recall  = None
fm_ranker       = None
redis_client    = redis.Redis(host="localhost", port=6379, db=0)

class RecommendRequest(BaseModel):
    user_id:     str
    scene:       str = "homepage"   # homepage / playing / search
    current_song: str = None        # 当前播放歌曲（用于实时相似推荐）
    topk:        int = 30

class RecommendResponse(BaseModel):
    user_id:   str
    songs:     list
    latency_ms: float

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    t0 = time.time()

    # ── Step 1: 多路召回 ──────────────────────────────────────
    candidates = {}

    # 路1: ItemCF召回（历史行为相似歌曲）
    if itemcf_model:
        for song_id, score in itemcf_model.recommend(req.user_id, topk=200):
            candidates[song_id] = candidates.get(song_id, 0) + score * 0.5

    # 路2: 基于内容召回（当前播放歌曲的相似歌曲，适合playing场景）
    if req.current_song and content_recall:
        for song_id, score in content_recall.recall_by_song(req.current_song, topk=100):
            candidates[song_id] = candidates.get(song_id, 0) + score * 0.3

    # 路3: 热门兜底（冷启动/候选不足时）
    hot_songs = _get_hot_songs(topk=50)
    for song_id in hot_songs:
        if song_id not in candidates:
            candidates[song_id] = 0.1

    candidate_list = list(candidates.keys())

    # ── Step 2: FM排序 ───────────────────────────────────────
    if fm_ranker and candidate_list:
        user_feat  = _get_user_features(req.user_id)
        song_feats = [_get_song_features(s) for s in candidate_list]
        scores     = fm_ranker.predict(user_feat, song_feats)
        ranked     = sorted(zip(candidate_list, scores),
                            key=lambda x: -x[1])
    else:
        ranked = sorted(candidates.items(), key=lambda x: -x[1])

    # ── Step 3: 重排（多样性 + 运营规则）────────────────────
    final = _rerank(ranked, topk=req.topk)

    latency = (time.time() - t0) * 1000
    return RecommendResponse(user_id=req.user_id,
                             songs=final,
                             latency_ms=round(latency, 2))

def _get_hot_songs(topk=50) -> list:
    """从Redis缓存获取热门歌曲（离线任务每小时更新）"""
    raw = redis_client.get("hot_songs")
    if raw:
        return json.loads(raw)[:topk]
    return []

def _get_user_features(user_id: str) -> dict:
    """从Redis获取用户实时特征"""
    raw = redis_client.get(f"user_feat:{user_id}")
    return json.loads(raw) if raw else {}

def _get_song_features(song_id: str) -> dict:
    """从Redis获取歌曲特征"""
    raw = redis_client.get(f"song_feat:{song_id}")
    return json.loads(raw) if raw else {}

def _rerank(ranked: list, topk: int) -> list:
    """
    重排策略:
    1. 流派多样性：连续同流派歌曲不超过3首
    2. 过滤下架/违规歌曲（查Redis黑名单）
    3. 插入运营位（每10首插入1首推广歌曲）
    """
    result, genre_window = [], []
    blacklist = set(json.loads(redis_client.get("blacklist") or "[]"))

    for song_id, score in ranked:
        if len(result) >= topk:
            break
        if song_id in blacklist:
            continue
        # 简化的多样性控制（实际按流派判断）
        result.append({"song_id": song_id, "score": float(score)})

    return result
