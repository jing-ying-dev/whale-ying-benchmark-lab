# -*- coding: utf-8 -*-
"""WhaleYing 记忆增强层 · 多跳 demo（SRL 拆题 + 逐跳作答，模型可换）。

让 3B 模型在 262k 多跳上打到 47% 的评测入口：
  SRL 拆题（route-1 拆链）→ 逐跳（BM25 检索 + 模型作答，上一步答案填下一步）。

用法：
  py -3 demo.py 3b            # 用 qwen2.5-coder:3b（Ollama 本地）
  py -3 demo.py gpt           # 用 gpt-4o-mini（OpenAI API）
  py -3 demo.py deepseek      # 用 deepseek-flash（DeepSeek API）
  py -3 demo.py 3b 1 20       # 指定题号
  py -3 demo.py 3b --live     # 现场调 DeepSeek 拆题（默认复用已拆好的数据）
"""
import io, sys, json, os, re
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 动态定位当前脚本目录，实现完全自包含运行
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

import our_system as OS
from openai import OpenAI

# ---------------- 模型注册表：换模型只改这里 ----------------
MODELS = {
    "3b":       dict(model="qwen2.5-coder:3b", temperature=0.0, max_tokens=10,
                     base_url="http://127.0.0.1:11434/v1", api_key="ollama"),
    "gpt":      dict(model="gpt-4o-mini", temperature=0.7, max_tokens=100,
                     api_key_env="OPENAI_API_KEY"),
    "deepseek": dict(model="deepseek-flash", temperature=0.7, max_tokens=100,
                     base_url="https://api.deepseek.com", api_key_env="DEEPSEEK_API_KEY"),
}

def make_agent(model_name):
    cfg = MODELS[model_name]

    class A:
        pass
    agent = A()
    agent.model = cfg["model"]
    agent.temperature = cfg["temperature"]
    agent.max_tokens = cfg["max_tokens"]
    agent.retrieve_num = 10
    agent.sub_dataset = "factconsolidation_mh_262k"
    agent.agent_name = "Agentic_memory_our_system"
    agent._retr_cache = None
    agent._emb_model = None
    agent._emb_failed = True          # BM25-only（与 47% 一致；向量+RRF 消融无增益）

    def _client(self, _cfg=cfg):
        kw = dict(max_retries=3, timeout=600.0)
        if "base_url" in _cfg:
            kw["base_url"] = _cfg["base_url"]
        if "api_key" in _cfg:
            kw["api_key"] = _cfg["api_key"]
        elif "api_key_env" in _cfg:
            kw["api_key"] = os.environ.get(_cfg["api_key_env"])
        return OpenAI(**kw)
    A._create_oai_client = _client
    return agent


# ---------------- 判分（归一化 exact match） ----------------
def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()

def eq(a, b):
    return norm(a) == norm(b)


# ---------------- 拆题链解析 ----------------
def steps_to_hops(steps):
    start = None
    rels = []
    for s in steps:
        m = re.search(r"return the (.+?) of #\d+", s)
        if m:
            rels.append(m.group(1).strip()); continue
        m2 = re.search(r"return the (.+?) of (.+)", s)
        if m2:
            if start is None:
                start = m2.group(2).strip(); rels.append(m2.group(1).strip())
            else:
                rels.append(m2.group(1).strip())
            continue
        m3 = re.match(r"return (.+)", s)
        if m3 and start is None:
            start = m3.group(1).strip()
    return start, rels


# ---------------- 单跳作答 ----------------
def single_hop_card(agent, query, rows, card_text, hop_no, total):
    order = OS._hybrid_retrieve(agent, query, rows)
    topk_rows = [rows[i] for i in order[:agent.retrieve_num]]
    message = ("QUESTION CARD (the multi-hop question decomposed into a chain of single-hop questions; "
               "「{上一步答案}」 means the previous step's answer fills this blank):\n%s\n\n"
               "Now Answer the Question (step %d of %d): %s\nAnswer:"
               % (card_text, hop_no, total, query))
    ans, _, _ = OS._direct_answer(agent, message, topk_rows)
    return (ans or "").strip(), topk_rows


# ---------------- 跑一题，返回完整轨迹 ----------------
def run_question(agent, e, rows):
    qid = e.get("qid")
    q = e["question"]
    gold = e.get("gold", "")
    steps = e.get("steps", [])
    trace = []
    start, rels = steps_to_hops(steps)
    if not start or not rels:
        return {"qid": qid, "question": q, "gold": gold, "ok": False,
                "final": None, "hops": [], "trace": trace, "note": "拆题失败"}
    hops = [(start, rels[0])] + [("", r) for r in rels[1:]]
    card_lines = []
    for i, (subj, rel) in enumerate(hops):
        tail = "  【最终答案：只写答案实体，禁止复述整句】" if i == len(hops) - 1 else ""
        card_lines.append("%d. %s of %s%s" % (i + 1, rel, subj if subj else "{上一步答案}", tail))
    card_text = "\n".join(card_lines)
    prev = None
    final = None
    for i, (subj, rel) in enumerate(hops):
        s = subj or prev
        query = "What is the %s of %s?" % (rel, s)
        ans, topk_rows = single_hop_card(agent, query, rows, card_text, i + 1, len(hops))
        trace.append({"hop": i + 1, "query": query, "ans": ans,
                      "top": [str(r.get("等式") or r.get("左值指标名") or "").strip()
                              for r in topk_rows[:3]]})
        prev = ans
        final = ans
    ok = final is not None and eq(final, gold)
    return {"qid": qid, "question": q, "gold": gold, "ok": ok,
                "final": final, "hops": hops, "trace": trace}


# ---------------- 现场拆题（可选，--live） ----------------
DECOMPOSE_PROMPT = """You are a linguistic expert specializing in Semantic Role Labeling (SRL).
Decompose the complex multi-hop question into sequential single-hop steps.
Allowed relations: {relations}

Question: {question}

Return JSON with "steps": ["return the <relation> of <entity>", "return the <relation> of #1", ...]"""

def live_decompose(question, relset):
    client = OpenAI(base_url="https://api.deepseek.com", api_key=os.environ.get("DEEPSEEK_API_KEY"))
    REL_STR = ", ".join(relset)
    prompt = DECOMPOSE_PROMPT.replace("{relations}", REL_STR).replace("{question}", question)
    r = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    return json.loads(r.choices[0].message.content)


def main():
    args = [a for a in sys.argv[1:]]
    model_name = args[0] if args and args[0] in MODELS else "3b"
    rest = args[1:]
    live = "--live" in rest
    qids = [int(a) for a in rest if a.isdigit()]
    if not qids:
        qids = [1, 20]          # 默认跑 1 和 20 两道经典题

    print("=" * 70)
    print("WhaleYing 记忆增强层 · 多跳 demo · 模型 = %s（%s）"
          % (model_name, MODELS[model_name]["model"]))
    print("=" * 70)

    # 载入数据（优先读取相对路径 ./data/）
    data_dir = BASE_DIR / "data"
    store_path = data_dir / "our_system_store_mh262k_new.json"
    if not store_path.exists():
        # 回退兼容
        store_path = BASE_DIR / "our_system_store_mh262k_new.json"
        
    rows = json.load(open(store_path, encoding="utf-8"))
    rows = OS._fold_latest(rows)
    print("Store: %d 条事实 -> Fold 最新状态后: %d 条" % (len(json.load(open(store_path, encoding="utf-8"))), len(rows)))

    route1_path = data_dir / "route1-100题.json" if (data_dir / "route1-100题.json").exists() else BASE_DIR / "route1-100题.json"
    relset_path = data_dir / "规范关系表-清洗.json" if (data_dir / "规范关系表-清洗.json").exists() else BASE_DIR / "规范关系表-清洗.json"

    route1 = json.load(open(route1_path, encoding="utf-8"))
    relset = set(json.load(open(relset_path, encoding="utf-8")))

    agent = make_agent(model_name)
    n_right = 0
    for qid in qids:
        e = route1.get(str(qid)) or route1.get(qid)
        if not e:
            print("\n[题 %d] 不在题目库里，跳过" % qid)
            continue
        if live:
            try:
                r = live_decompose(e["question"], relset)
                e = dict(e, steps=r.get("steps", []), srl=r.get("srl", []))
            except Exception as ex:
                print("[题 %d] 现场拆题失败，退回已拆好的数据：%s" % (qid, str(ex)[:80]))
        res = run_question(agent, e, rows)
        n_right += 1 if res["ok"] else 0
        print("\n" + "-" * 70)
        print("题 %s: %s" % (qid, res["question"]))
        print("Gold 答案: %s" % res["gold"])
        print("拆题链: " + " -> ".join("the %s of %s" % (r, s if s else "#上一步")
                                      for s, r in res["hops"]))
        for t in res["trace"]:
            print("  [第 %d 跳] %s" % (t["hop"], t["query"]))
            for k, fact in enumerate(t["top"], 1):
                print("      检索 #%d: %s" % (k, fact))
            print("      => 答: %s" % t["ans"])
        print("最终答案: %s" % res["final"])
        print("判定: %s" % ("✓ 对" if res["ok"] else "✗ 错 (Gold=%s)" % res["gold"]))
    print("\n" + "=" * 70)
    print("本次 %d 题，对 %d（%.0f%%）" % (len(qids), n_right, 100.0 * n_right / max(1, len(qids))))
    print("=" * 70)


if __name__ == "__main__":
    main()
