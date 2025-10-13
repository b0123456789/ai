from modelscope import snapshot_download

# 下载 BAAI/bge-large-zh 到本地目录
model_dir = snapshot_download(
    model_id="BAAI/bge-large-zh",  # 模型 ID
    cache_dir="e:/code/local_bge_large_zh",  # 本地保存路径
    revision="master"  # 版本（可选，默认最新）
)
