# Third-Party Notices

## MemoryAgentBench

WhaleYing Benchmark Lab uses benchmark data derived from **MemoryAgentBench** for evaluating memory capabilities and multi-hop reasoning in LLM agents.

### Attribution

- **Project:** MemoryAgentBench
- **Paper:** [Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions](https://arxiv.org/abs/2507.05257) (ICLR 2026, Hu, Wang, McAuley)
- **Official repository:** [https://github.com/HUST-AI-HYZ/MemoryAgentBench](https://github.com/HUST-AI-HYZ/MemoryAgentBench)
- **Hugging Face dataset:** [https://huggingface.co/datasets/ai-hyz/MemoryAgentBench](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench)
- **Declared dataset license:** MIT

### Dataset provenance

The experiments use the following upstream dataset revision:

- **Dataset repository revision:** `7ea066982b140a19337e17e60d45d4076e042faf`
- **Conflict_Resolution data-file revision:** `00d1946269e29b41eed74511997afa8171b91e08`
- **File:** `Conflict_Resolution-00000-of-00001.parquet`
- **SHA256:** `24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45`

The local Hugging Face cache and experiment logs identify the dataset repository revision used during the original experiments. The Parquet file's SHA256 matches the corresponding upstream Git LFS object.

WhaleYing Benchmark Lab uses pre-extracted data artifacts generated from this dataset through the upstream WhaleYing Memory project. It does not independently download the original dataset during the benchmark demo.

### License clarification

The MIT license applies to the respective upstream MemoryAgentBench materials. The Apache-2.0 license in this repository governs WhaleYing Benchmark Lab's own code and does not replace or modify the upstream dataset's license.

The original benchmark data remains attributed to its upstream authors. Our extraction pipeline, memory configurations, evaluation harness, experimental results, and analysis are project contributions.

### Dataset hierarchy and task identification

MemoryAgentBench uses `Conflict_Resolution` as a top-level Hugging Face dataset split. Within this split, this project selects the `FactConsolidation` task through the `metadata.source` field, specifically `factconsolidation_mh_262k` for the 262K multi-hop evaluation.

In other words:

- Dataset split: `Conflict_Resolution`
- Task source: `factconsolidation_mh_262k`
- Evaluation setting: multi-hop, 262K context

The terms `Conflict_Resolution` and `FactConsolidation` refer to different levels of the dataset hierarchy, not different evaluation tasks.
