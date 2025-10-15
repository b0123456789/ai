import os
import torch
import faiss
import json  # 新增：导入json模块
from flask import Flask, request, make_response  # 新增：导入make_response
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from flask_cors import CORS  # 导入扩展

app = Flask(__name__)
CORS(app)  # 全局启用 CORS，允许所有源

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
        json.dumps({"status":  True, "message": success_msg, "data": data},
                   ensure_ascii=False),
        200,
        {'Content-Type': 'application/json; charset=utf-8'}
    )


def jsonErr(error_msg):
    return make_response(
        json.dumps({"status": False, "message": error_msg},
                   ensure_ascii=False),
        400,
        {'Content-Type': 'application/json; charset=utf-8'}
    )


@app.route('/list/vectorstore/doc', methods=['GET'])
def list_vectorstore_doc():
    page = int(request.args.get('page', '1'))
    print(page)
    return jsonOk("查询成功", get_paginated_data(page))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
