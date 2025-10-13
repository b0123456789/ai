import os
import torch
import faiss
import json  # 新增：导入json模块
from flask import Flask, request, make_response  # 新增：导入make_response
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from modelscope import snapshot_download, AutoTokenizer, AutoModel
from transformers import TextStreamer, AutoModelForCausalLM, pipeline
from threading import Lock
from langchain.prompts.prompt import PromptTemplate

# --------------------------
# 全局配置
# --------------------------
model_dir = snapshot_download(
    model_id='deepseek-ai/deepseek-llm-7b-chat',
    cache_dir='e:/code/deepseek-7b-chat',
    revision='master'
)
BGE_EMBEDDING_PATH = "E:/code/local_bge_large_zh/BAAI/bge-large-zh"
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.9
LLM_MAX_NEW_TOKENS = 512

# --------------------------
# 初始化Flask与全局资源
# --------------------------
app = Flask(__name__)
vectorstore_lock = Lock()

# 1. 加载大模型（LLM）（不变）
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)
llm_model.config.temperature = LLM_TEMPERATURE
llm_model.config.top_p = LLM_TOP_P
llm_model.config.max_new_tokens = LLM_MAX_NEW_TOKENS

transformers_pipeline = pipeline(
    task="text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    trust_remote_code=True
)
llm = HuggingFacePipeline(pipeline=transformers_pipeline)

# 2. 加载嵌入模型（不变）
if not os.path.exists(BGE_EMBEDDING_PATH):
    raise ValueError(f"嵌入模型路径不存在：{BGE_EMBEDDING_PATH}")
embeddings_model = HuggingFaceEmbeddings(
    model_name=BGE_EMBEDDING_PATH,
    model_kwargs={"device": "cuda" if torch.cuda.is_available()
                  else "cpu", "trust_remote_code": True}
    # model_kwargs={"device": "meta", "trust_remote_code": True}
)

# --------------------------
# 关键修正：FAISS初始化仅需index和embedding
# --------------------------
# 动态获取嵌入维度（不变）
test_text = "test"
try:
    test_embedding = embeddings_model.embed_query(test_text)
    embedding_dim = len(test_embedding)
except Exception as e:
    raise ValueError(f"无法获取嵌入维度：{str(e)}") from e

# 创建空的FAISS索引（不变）
empty_index = faiss.IndexFlatL2(embedding_dim)

# 初始化FAISS向量库：**仅传入index和embedding**
knowledge_data = [
    {"source": "柔嘉", "content": "柔嘉"},
]
texts = [item["content"] for item in knowledge_data]
metadatas = [{"source": item["source"]} for item in knowledge_data]
vectorstore = FAISS.from_texts(
    texts=texts,
    embedding=embeddings_model,
    metadatas=metadatas
)

# 初始化QA链（关联空向量库，不变）
# 初始化QA链（修正：自定义Prompt模板去掉英文前缀）

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    # 关键修改：通过chain_type_kwargs传递自定义Prompt模板
    chain_type_kwargs={
        "prompt": PromptTemplate(
            # 必须匹配StuffDocumentsChain的输入变量
            input_variables=["context", "question"],
            template='''{context}
Question: {question}
Answer:'''  # 自定义模板：无英文前缀
        )
    }
)

# --------------------------
# 接口1：构建/更新知识库（替换jsonify为json.dumps）
# --------------------------


@app.route('/build_knowledge_base', methods=['POST'])
def build_knowledge_base():
    try:
        data = request.get_json()
        if not data or 'knowledge_data' not in data or not isinstance(data['knowledge_data'], list):
            # 错误响应：用json.dumps+make_response
            error_msg = {"status": "error", "message": "缺少'knowledge_data'列表"}
            return make_response(
                json.dumps(error_msg, ensure_ascii=False),
                400,
                {'Content-Type': 'application/json; charset=utf-8'}
            )

        knowledge_data = data['knowledge_data']
        for item in knowledge_data:
            if 'source' not in item or 'content' not in item:
                error_msg = {"status": "error",
                             "message": "每个条目需包含'source'和'content'"}
                return make_response(
                    json.dumps(error_msg, ensure_ascii=False),
                    400,
                    {'Content-Type': 'application/json; charset=utf-8'}
                )

        texts = [item['content'] for item in knowledge_data]
        metadatas = [{"source": item['source']} for item in knowledge_data]

        with vectorstore_lock:
            global vectorstore
            vectorstore.add_texts(texts=texts, metadatas=metadatas)

        # 成功响应：用json.dumps+make_response
        success_msg = {"status": "success",
                       "message": f"知识库更新完成，共添加{len(texts)}条数据"}
        return make_response(
            json.dumps(success_msg, ensure_ascii=False),
            200,
            {'Content-Type': 'application/json; charset=utf-8'}
        )

    except Exception as e:
        error_msg = {"status": "error", "message": f"构建失败：{str(e)}"}
        return make_response(
            json.dumps(error_msg, ensure_ascii=False),
            500,
            {'Content-Type': 'application/json; charset=utf-8'}
        )

# --------------------------
# 接口2：知识库查询（替换jsonify为json.dumps）
# --------------------------


@app.route('/query', methods=['POST'])
def query_knowledge_base():
    try:
        data = request.get_json()
        if not data or 'query' not in data or not isinstance(data['query'], str) or not data['query'].strip():
            error_msg = {"status": "error", "message": "缺少有效'query'字段"}
            return make_response(
                json.dumps(error_msg, ensure_ascii=False),
                400,
                {'Content-Type': 'application/json; charset=utf-8'}
            )

        query = data['query'].strip()
        if vectorstore.index.ntotal == 0:
            error_msg = {"status": "error",
                         "message": "知识库为空，请先调用/build_knowledge_base"}
            return make_response(
                json.dumps(error_msg, ensure_ascii=False),
                400,
                {'Content-Type': 'application/json; charset=utf-8'}
            )

        result = qa_chain.invoke({"query": query})

        formatted_result = {
            "question": query,
            "answer": result["result"].strip(),
            "sources": [
                {
                    "source": doc.metadata["source"],
                    "content": doc.page_content.strip()
                }
                for doc in result["source_documents"]
            ]
        }

        # 成功响应：用json.dumps+make_response
        return make_response(
            json.dumps(formatted_result, ensure_ascii=False),
            200,
            {'Content-Type': 'application/json; charset=utf-8'}
        )

    except Exception as e:
        error_msg = {"status": "error", "message": f"查询失败：{str(e)}"}
        return make_response(
            json.dumps(error_msg, ensure_ascii=False),
            500,
            {'Content-Type': 'application/json; charset=utf-8'}
        )


# --------------------------
# 启动应用（不变）
# --------------------------
if __name__ == '__main__':
    try:
        qa_chain.invoke({"query": "Warm up"})
        app.logger.info("服务预热成功")
    except Exception as e:
        app.logger.error(f"预热失败：{str(e)}")

    app.run(host='0.0.0.0', port=5000, debug=False)
