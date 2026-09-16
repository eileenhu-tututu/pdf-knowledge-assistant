import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import APIConnectionError, AuthenticationError, OpenAI


# 固定读取脚本同目录的 .env，避免 VS Code 工作目录不同导致配置丢失。
env_file = Path(__file__).resolve().parent / ".env"
if not env_file.is_file():
    raise FileNotFoundError(f"未找到环境变量文件：{env_file}")

load_dotenv(dotenv_path=env_file, override=True)

api_key = (os.getenv("SILICONFLOW_API_KEY") or "").strip()
base_url = (os.getenv("SILICONFLOW_BASE_URL") or "").strip()
model = (os.getenv("LLM_MODEL") or "").strip()

missing = [
    name
    for name, value in {
        "SILICONFLOW_API_KEY": api_key,
        "SILICONFLOW_BASE_URL": base_url,
        "LLM_MODEL": model,
    }.items()
    if not value
]
if missing:
    raise RuntimeError(f".env 中缺少或未正确填写：{', '.join(missing)}")

masked_key = f"{api_key[:5]}...{api_key[-4:]}" if len(api_key) >= 10 else "<格式异常>"
print(f"已加载配置：key={masked_key}, base_url={base_url}, model={model}")

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    # 避免 VS Code 或系统中的无效代理变量干扰请求。
    http_client=httpx.Client(trust_env=False, timeout=30.0),
)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "请用一句话解释什么是RAG。",
            }
        ],
    )
except AuthenticationError as exc:
    raise SystemExit("认证失败：请重新生成 SiliconFlow API Key，并更新 .env。") from exc
except APIConnectionError as exc:
    raise SystemExit(
        f"配置已成功读取，但无法连接 {base_url}。请检查 DNS、VPN、代理或当前网络。"
    ) from exc

print(response.choices[0].message.content)
