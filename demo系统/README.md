# WhaleYing 记忆增强层 · 多跳 demo

这是让 **Qwen2.5-Coder-3B 在 262k 多跳上打到 47%** 的那条管线的落成入口，模型可换。

## 管线（三步）

1. **抽取**：DeepSeek 把 262k 原文抽成四表（facts/metrics/cards/identities），存 store（`tmp\our_system_store_mh262k_new.json`，已抽好）。
2. **SRL 拆题（route-1）**：DeepSeek 把多跳题拆成有序单跳链（`tmp\route1-100题.json`，已拆好 100 题；`--live` 可现场重拆）。
3. **逐跳作答**：BM25 检索 → 模型答当前跳 → 上一步答案填下一步 → 末跳即最终答案。

## 怎么跑

在 `<工作目录>\tmp\demo系统` 下：

```powershell
$env:PYTHONIOENCODING='utf-8'
py -3 demo.py 3b          # qwen2.5-coder:3b（Ollama 本地，需先起 Ollama）
py -3 demo.py gpt         # gpt-4o-mini（走 .env 的 OPENAI_API_KEY）
py -3 demo.py deepseek    # deepseek-flash（走 .env 的 DEEPSEEK_API_KEY）
py -3 demo.py 3b 0 5 12   # 指定题号（0 起，最多 99）
py -3 demo.py 3b --live   # 现场调 DeepSeek 拆题（默认复用已拆好的）
```

## 换模型 = 改一处

`demo.py` 顶部的 `MODELS` 字典，加一行就行：

```python
MODELS = {
    "3b":       dict(model="qwen2.5-coder:3b", base_url="http://127.0.0.1:11434/v1", api_key="ollama", ...),
    "gpt":      dict(model="gpt-4o-mini", api_key_env="OPENAI_API_KEY", ...),
    "deepseek": dict(model="deepseek-flash", base_url="https://api.deepseek.com", api_key_env="DEEPSEEK_API_KEY", ...),
    # 加一个：任意 OpenAI 兼容端点
    "我的模型": dict(model="...", base_url="...", api_key="...", ...),
}
```

## 判分口径（诚实）

逐跳作答的判分是**归一化 exact match**（`norm(a) == norm(b)`，小写 + 去非字母数字），**不是** substring。所以 47% 是「严」口径，substring 只会更高。

## 依赖（已在本机，无需新装）

- 核心模块：`C:\Users\28163\聚合测试\l1test\MemoryAgentBench\methods\our_system.py`（及它依赖的 `抽取器\`、`utils\`）。
- 数据：`tmp\our_system_store_mh262k_new.json`、`tmp\route1-100题.json`、`tmp\规范关系表-清洗.json`。
- 3b 需要 Ollama 起在 `127.0.0.1:11434`（模型 `qwen2.5-coder:3b`）。
