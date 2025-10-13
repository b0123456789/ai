import os
import torch
from langchain_community.vectorstores import FAISS
# -------------------------- 修正：导入 langchain-huggingface 的 Embeddings --------------------------
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from modelscope import snapshot_download, AutoTokenizer, AutoModel
from transformers import TextStreamer, AutoModelForCausalLM, pipeline


# --------------------------
# 1. 加载 DeepSeek 模型（本地缓存）并配置生成参数
# --------------------------
model_dir = snapshot_download(
    model_id='deepseek-ai/deepseek-llm-7b-chat',
    cache_dir='e:/code/deepseek-7b-chat',
    revision='master'
)

tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

# 将生成参数写入模型 config
model.config.temperature = 0.7
model.config.top_p = 0.9
model.config.max_new_tokens = 512


# --------------------------
# 2. 创建 Transformers Pipeline（含流式输出）
# --------------------------
streamer = TextStreamer(
    tokenizer=tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

transformers_pipeline = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    streamer=streamer
)

# 初始化 LangChain 兼容的 LLM
llm = HuggingFacePipeline(pipeline=transformers_pipeline)


# --------------------------
# 3. 准备知识库数据
# --------------------------
knowledge_data = [
    {"source": "文档1", "content": "DeepSeek 是由深度求索（DeepSeek AI）开发的开源大语言模型"},
    {"source": "文档2", "content": "DeepSeek-LLM 支持 Chat、Code、Embedding 等多种任务"},
    {"source": "文档3", "content": "RAG 技术（检索增强生成）通过外部知识库解决大模型的幻觉问题"},
    {"source": "文档4", "content": "本知识库用于测试 LangChain 的 RAG 功能"}
]
texts = [item["content"] for item in knowledge_data]
metadatas = [{"source": item["source"]} for item in knowledge_data]


# --------------------------
# 4. 加载本地 BGE 嵌入模型（修正后）
# --------------------------
# local_embedding_model_path = "e:/code/local_bge_large_zh"
local_embedding_model_path = "E:/code/local_bge_large_zh/BAAI/bge-large-zh"
if not os.path.exists(local_embedding_model_path):
    raise ValueError(f"本地嵌入模型路径不存在：{local_embedding_model_path}")

# -------------------------- 修正：添加 trust_remote_code=True --------------------------
embeddings_model = HuggingFaceEmbeddings(
    model_name=local_embedding_model_path,
    model_kwargs={
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "trust_remote_code": True  # 信任模型自定义逻辑
    }
)


# --------------------------
# 5. 构建向量库与 RAG 链
# --------------------------
vectorstore = FAISS.from_texts(
    texts=texts,
    embedding=embeddings_model,
    metadatas=metadatas
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)


# --------------------------
# 6. 测试查询
# --------------------------
#query = "DeepSeek 是哪家公司开发的？"
query = "百度老板是谁"
result = qa_chain({"query": query})

print("问题:", query)
print("答案:", result["result"].strip())
print("来源文档:")
for doc in result["source_documents"]:
    print(f"- 来自 {doc.metadata['source']}: {doc.page_content.strip()}")
