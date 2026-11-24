# WhaleYing 记忆增强层 · 多跳 Demo

这是让 **Qwen2.5-Coder-3B 在 262k 多跳上达到 47%**（Normalized Exact Match）的可复现运行入口，模型支持一键替换。

## 管线架构（三步链条）

1. **抽取（Extraction）**：预先由 DeepSeek 将 262k 极端长文本抽取为四表结构（facts/cards/metrics/identities），存储于本地结构化记忆库（`data/our_system_store_mh262k_new.json`）。
2. **SRL 拆题（Decomposition）**：通过语义角色标注（SRL route-1）将复杂多跳问题拆解为有序单跳链（内置 100 题基准于 `data/route1-100题.json`；支持 `--live` 参数现场调用 API 重拆）。
3. **逐跳作答（Step-by-Step Answering）**：BM25 检索记忆证据 → 模型回答当前跳子问题 → 前序答案填入后序槽位 → 最后一跳输出归一化实体答案。

## 快速运行

在当前 `demo系统/` 目录下执行：

```powershell
$env:PYTHONIOENCODING='utf-8'

# 1. 运行本地端侧 3B 模型（默认评测第 1 题和第 20 题，需预先启动 Ollama）
py -3 demo.py 3b

# 2. 指定题目批量评测（如第 1、6、20、52、55、13 题）
py -3 demo.py 3b 1 6 20 52 55 13

# 3. 评测云端商用模型（需配置环境变量 OPENAI_API_KEY / DEEPSEEK_API_KEY）
py -3 demo.py gpt 1 20
py -3 demo.py deepseek 1 20

# 4. 现场动态拆题评测（通过 DeepSeek API 现场生成单跳链）
py -3 demo.py 3b 1 --live
```

## 切换模型

在 `demo.py` 顶部的 `MODELS` 注册表中添加一行配置即可支持任意兼容端点：

```python
MODELS = {
    "3b":       dict(model="qwen2.5-coder:3b", base_url="http://127.0.0.1:11434/v1", api_key="ollama"),
    "gpt":      dict(model="gpt-4o-mini", api_key_env="OPENAI_API_KEY"),
    "deepseek": dict(model="deepseek-flash", base_url="https://api.deepseek.com", api_key_env="DEEPSEEK_API_KEY"),
    # 自定义模型接入
    "custom":   dict(model="your-model-name", base_url="http://localhost:8000/v1", api_key="sk-..."),
}
```

## 评分标准说明

逐跳作答采用**严格归一化精确匹配**（`norm(a) == norm(b)`，统一转小写、去除非字母数字字符、去除首尾空白），要求输出实体与金标完全相等，**严于** 宽泛的 `substring_exact_match` 子串命中。

## 环境与依赖

- **核心算法**：当前目录下自包含的 `our_system.py`（折叠、混合检索、一步作答）。
- **评测数据**：`data/our_system_store_mh262k_new.json`（8.9 MB 记忆库）、`data/route1-100题.json`、`data/规范关系表-清洗.json`。
- **本地模型环境**：运行 3B 模型需本地起好 Ollama（`ollama run qwen2.5-coder:3b`）。
- **Python 依赖**：`rank-bm25`, `openai`, `python-dotenv`。
