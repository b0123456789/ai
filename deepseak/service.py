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
import mysql.connector
from mysql.connector import Error
from datetime import datetime

# --------------------------
# 全局配置
# --------------------------
model_dir = snapshot_download(
    model_id='deepseek-ai/deepseek-llm-7b-chat',
    cache_dir='e:/code/deepseek-7b-chat',
    revision='master'
)
VECTORSTORE_PATH = "./faiss_index"  # 新增：FAISS本地保存路径
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
try:
    test_text = "test"
    test_embedding = embeddings_model.embed_query(test_text)
    embedding_dim = len(test_embedding)
except Exception as e:
    raise ValueError(f"无法获取嵌入维度：{str(e)}") from e

# 创建空的FAISS索引（不变）
# empty_index = faiss.IndexFlatL2(embedding_dim)

# 初始化FAISS向量库：**仅传入index和embedding**


if os.path.exists(VECTORSTORE_PATH):
    app.logger.info("正在加载本地FAISS索引...")
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings=embeddings_model,
        allow_dangerous_deserialization=True  # 必须添加
    )
else:
    knowledge_data = [{"source": "", "content": ""},]
    texts = [item["content"] for item in knowledge_data]
    metadatas = [{"source": item["source"]} for item in knowledge_data]
    app.logger.info("未找到本地FAISS索引，创建新的...")
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
# mysql
# --------------------------

conn = mysql.connector.connect(
    host='localhost',      # 数据库地址（本地/远程）
    user='root',           # 用户名
    password='root',  # 密码
    database='doc',    # 要操作的数据库（需提前创建）
    charset='utf8mb4',     # 支持中文/emoji
    port=3306              # MySQL默认端口
)

if not conn.is_connected():
    print("mysql 没有连接")
    os._exit(0)


def insert_data(source, content):
    remove_data(source)
    cursor = conn.cursor()
    try:
        sql = "INSERT INTO doc (source, content,createtime) VALUES (%s, %s, %s)"
        now = datetime.now()
        cursor.execute(
            sql,  (source, content, now.strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()  # 提交事务（增删改必须执行）
    except Error as e:
        conn.rollback()  # 出错回滚
    finally:
        cursor.close()


def remove_data(source):
    cursor = conn.cursor()
    try:
        sql = "DELETE FROM doc WHERE source = %s"
        cursor.execute(sql, (source,))
        conn.commit()
    except Error as e:
        conn.rollback()
    finally:
        cursor.close()


def get_paginated_data(page: int = 1, per_page: int = 20):
    try:
        cursor = conn.cursor(dictionary=True)  # 返回字典格式结果

        # 计算偏移量
        if page - 1 < 0:
            page = 1

        offset = (page - 1) * per_page

        # 1. 查询当前页数据
        query = "SELECT * FROM doc ORDER BY id DESC LIMIT %s OFFSET %s"
        cursor.execute(query, (per_page, offset))
        data = cursor.fetchall()

        # 2. 查询总记录数
        cursor.execute("SELECT COUNT(*) AS total FROM doc")
        total = cursor.fetchone()['total']

        # 计算总页数
        total_pages = (total + per_page - 1) // per_page

        for row in data:
            row["createtime"] = row["createtime"].strftime("%Y-%m-%d %H:%M:%S")

        return {
            "row": data,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        }
    except Error as e:
        print(f"数据库错误: {e}")
        return None
    finally:
        cursor.close()


def jsonOk(success_msg, data={}):
    return make_response(
        json.dumps({"status": "error", "message": success_msg, "data": data},
                   ensure_ascii=False),
        200,
        {'Content-Type': 'application/json; charset=utf-8'}
    )


def jsonErr(error_msg):
    return make_response(
        json.dumps({"status": "success", "message": error_msg},
                   ensure_ascii=False),
        400,
        {'Content-Type': 'application/json; charset=utf-8'}
    )

# --------------------------
# 接口1：构建/更新知识库）
# --------------------------


@app.route('/build_knowledge_base', methods=['POST'])
def build_knowledge_base():
    try:
        data = request.get_json()
        if not data or 'knowledge_data' not in data or not isinstance(data['knowledge_data'], list):
            return jsonErr("缺少'knowledge_data'列表")

        knowledge_data = data['knowledge_data']
        for item in knowledge_data:
            if 'source' not in item or 'content' not in item:
                return jsonErr("每个条目需包含'source'和'content'")

        texts = [item['content'] for item in knowledge_data]
        metadatas = [{"source": item['source']} for item in knowledge_data]

        # 写入mysql
        for item in knowledge_data:
            insert_data(item["source"], item["content"])

        with vectorstore_lock:
            global vectorstore
            vectorstore.add_texts(texts=texts, metadatas=metadatas)

        return jsonOk(f"知识库更新完成，共添加{len(texts)}条数据")

    except Exception as e:
        return jsonErr(f"构建失败：{str(e)}")

# --------------------------
# 接口2：知识库查询）
# --------------------------


@app.route('/query', methods=['POST'])
def query_knowledge_base():
    try:
        data = request.get_json()
        if not data or 'query' not in data or not isinstance(data['query'], str) or not data['query'].strip():
            return jsonErr("缺少有效'query'字段")

        query = data['query'].strip()
        if vectorstore.index.ntotal == 0:
            return jsonErr("知识库为空，请先调用/build_knowledge_base")

        result = qa_chain.invoke({"query": query})

        formatted_result = {
            "question": query,
            "answer": result["result"].strip(),
            "sources": [
                {
                    "source": doc.metadata["source"],
                    "content": doc.page_content.strip()
                }
                for doc in result["source_documents"] if doc.metadata["source"] != "" and doc.page_content.strip() != ""
            ]
        }

        return jsonOk("查询成功", formatted_result)

    except Exception as e:
        return jsonErr(f"查询失败：{str(e)}")

# --------------------------
# 接口3：知识库保存
# --------------------------


@app.route('/save/vectorstore', methods=['POST'])
def save_vectorstore():
    vectorstore.save_local(VECTORSTORE_PATH)
    return jsonOk(f"知识库已保存到{VECTORSTORE_PATH}")

# --------------------------
# 接口4：知识库删除文档
# --------------------------


@app.route('/remove/vectorstore/doc', methods=['POST'])
def remove_vectorstore_doc():

    data = request.get_json()
    if not data or 'doc' not in data or not isinstance(data['doc'], str) or not data['doc'].strip():
        return jsonErr("缺少有效'doc'字段")

    doc = data['doc'].strip()

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": {"source": doc}  # 用 metadata 中的 source 字段过滤
        }
    )

    # 2. 获取符合条件的文档（查询内容可为空，filter 会生效）
    docs_to_delete = retriever.get_relevant_documents("")
    if docs_to_delete:
        # 提取文档ID列表
        ids_to_delete = [doc.id for doc in docs_to_delete]
        # 执行删除
        vectorstore.delete(ids=ids_to_delete)

    # mysql 删除文档
    remove_data(doc)

    return jsonOk(f"已删除文档{doc}")

# --------------------------
# 接口5：列出知识库文档
# --------------------------


@app.route('/list/vectorstore/doc', methods=['GET'])
def list_vectorstore_doc():
    page = int(request.args.get('page', '1'))
    print(page)
    return jsonOk("查询成功", get_paginated_data(page))


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
