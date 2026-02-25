# src/data_pipeline/hdf5_reader.py
# MSD每首歌存储为一个HDF5文件，包含55个字段
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

# MSD HDF5中的核心字段映射
SONG_FIELDS = {
    "metadata": ["title", "artist_name", "release", "year"],
    "analysis": ["tempo", "loudness", "duration", "key", "mode",
                 "time_signature", "danceability", "energy",
                 "segments_timbre",   # MFCC-like音色特征 (N×12矩阵)
                 "segments_pitches",  # 音高特征 (N×12矩阵)
    ],
    "musicbrainz": ["artist_mbid"],
    "tags": ["artist_terms", "artist_terms_freq"],  # 流派/风格标签
}

def read_song_hdf5(filepath: str) -> dict:
    """读取单首歌曲的HDF5文件，返回特征字典"""
    song = {}
    with h5py.File(filepath, "r") as f:
        # 基础元数据
        song["song_id"]     = f["metadata/songs"]["song_id"][0].decode()
        song["track_id"]    = f["metadata/songs"]["track_id"][0].decode()
        song["artist_name"] = f["metadata/songs"]["artist_name"][0].decode()
        song["title"]       = f["metadata/songs"]["title"][0].decode()
        song["year"]        = int(f["musicbrainz/songs"]["year"][0])
        # 音频特征（标量）
        song["tempo"]       = float(f["analysis/songs"]["tempo"][0])
        song["loudness"]    = float(f["analysis/songs"]["loudness"][0])
        song["duration"]    = float(f["analysis/songs"]["duration"][0])
        song["key"]         = int(f["analysis/songs"]["key"][0])
        song["mode"]        = int(f["analysis/songs"]["mode"][0])       # 大调/小调
        song["danceability"]= float(f["analysis/songs"]["danceability"][0])
        song["energy"]      = float(f["analysis/songs"]["energy"][0])
        # 音色特征聚合（对N个segment取均值/标准差，降维为固定长度向量）
        timbre = f["analysis/segments_timbre"][:]        # shape: (N, 12)
        song["timbre_mean"] = timbre.mean(axis=0).tolist()
        song["timbre_std"]  = timbre.std(axis=0).tolist()
        # 风格标签
        terms = f["metadata/artist_terms"][:]
        song["artist_terms"] = [t.decode() for t in terms]
    return song

def load_all_songs(data_dir: str, limit: int = None) -> pd.DataFrame:
    """批量读取所有HDF5文件，构建歌曲特征DataFrame"""
    records = []
    paths = list(Path(data_dir).rglob("*.h5"))
    if limit:
        paths = paths[:limit]
    for p in paths:
        try:
            records.append(read_song_hdf5(str(p)))
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")
    return pd.DataFrame(records)

def load_tasteprofile(filepath: str) -> pd.DataFrame:
    """
    读取用户播放记录 Taste Profile
    格式: user_id  song_id  play_count (TSV, 48M行)
    """
    df = pd.read_csv(filepath, sep="\t",
                     names=["user_id", "song_id", "play_count"])
    return df


# ─────────────────────────────────────────────────────────────
# src/data_pipeline/feature_engineer.py
# ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def build_song_feature_vector(df_songs: pd.DataFrame) -> pd.DataFrame:
    """
    从歌曲元数据构建内容特征向量
    用于基于内容的召回 & 排序模型的item侧特征
    """
    feat = df_songs.copy()
    # 1. 标量特征标准化
    scalar_cols = ["tempo", "loudness", "duration", "danceability", "energy"]
    scaler = MinMaxScaler()
    feat[scalar_cols] = scaler.fit_transform(feat[scalar_cols].fillna(0))

    # 2. 年代分桶（冷启动/内容推荐有用）
    feat["decade"] = (feat["year"] // 10 * 10).astype(str)

    # 3. 展开timbre_mean（12维音色均值）作为稠密特征
    timbre_df = pd.DataFrame(feat["timbre_mean"].tolist(),
                             columns=[f"timbre_{i}" for i in range(12)])
    feat = pd.concat([feat.drop(columns=["timbre_mean","timbre_std"]), timbre_df], axis=1)

    return feat

def build_user_feature(df_plays: pd.DataFrame,
                       df_songs: pd.DataFrame) -> pd.DataFrame:
    """
    从播放记录构建用户行为特征
    用于排序模型的user侧特征 & UserCF
    """
    # 播放次数作为隐式反馈权重（对数平滑）
    df = df_plays.copy()
    df["log_play"] = np.log1p(df["play_count"])

    # 用户级统计特征
    user_stats = df.groupby("user_id").agg(
        total_plays   = ("play_count", "sum"),
        unique_songs  = ("song_id",    "nunique"),
        avg_play      = ("play_count", "mean"),
        max_play      = ("play_count", "max"),
    ).reset_index()

    # 用户最喜欢的流派（通过歌曲-流派关联）
    df_m = df.merge(df_songs[["song_id", "artist_terms"]], on="song_id", how="left")
    # 展开标签，加权统计
    df_m = df_m.explode("artist_terms")
    user_genre = (df_m.groupby(["user_id", "artist_terms"])["log_play"]
                       .sum().reset_index()
                       .sort_values("log_play", ascending=False)
                       .groupby("user_id").head(5))  # top-5 流派偏好

    return user_stats, user_genre

def build_negative_samples(df_plays: pd.DataFrame,
                            all_song_ids: list,
                            neg_ratio: int = 1) -> pd.DataFrame:
    """
    正负样本构造：
    正样本 = 用户播放过的歌曲（label=1）
    负样本 = 用户未播放的歌曲中随机采样（label=0）
    保证正负比例为 1:neg_ratio
    """
    records = []
    user_songs = df_plays.groupby("user_id")["song_id"].apply(set).to_dict()
    song_pool  = set(all_song_ids)

    for uid, pos_set in user_songs.items():
        # 正样本
        for sid in pos_set:
            records.append({"user_id": uid, "song_id": sid, "label": 1})
        # 负样本
        neg_pool = list(song_pool - pos_set)
        n_neg    = min(len(pos_set) * neg_ratio, len(neg_pool))
        neg_samples = np.random.choice(neg_pool, n_neg, replace=False)
        for sid in neg_samples:
            records.append({"user_id": uid, "song_id": sid, "label": 0})

    return pd.DataFrame(records)
