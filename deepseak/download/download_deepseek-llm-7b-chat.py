from modelscope import snapshot_download, AutoModel, AutoTokenizer
from modelscope import GenerationConfig
model_dir = snapshot_download('deepseek-ai/deepseek-llm-7b-chat', cache_dir='e:/code/deepseek-7b-chat', revision='master')

#https://blog.csdn.net/m0_61664470/article/details/146039148