import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
from transformers import GenerationConfig

# 下载模型
model_dir = snapshot_download('deepseek-ai/deepseek-llm-7b-chat',
                              cache_dir='e:/code/deepseek-7b-chat',
                              revision='master')

# 加载tokenizer和模型 - 使用正确的模型类
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# 关键修改：使用AutoModelForCausalLM而不是AutoModel
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    model_dir, 
    trust_remote_code=True, 
    torch_dtype=torch.float16,
    device_map="auto"  # 自动分配到可用设备
)

# 设置padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 创建生成配置
generation_config = GenerationConfig(
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

# 正确的文本生成函数
def chat_with_model(prompt):
    # 格式化聊天消息
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 应用聊天模板
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # 生成响应
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            attention_mask=torch.ones_like(input_ids)
        )
    
    # 解码响应（跳过提示部分）
    response = tokenizer.decode(
        outputs[0][input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    return response

# 测试模型
try:
    prompt = "你好，最近怎么样？"
    response = chat_with_model(prompt)
    print("模型回复:", response)
except Exception as e:
    print(f"错误: {e}")