# Awesome-LLM-Pruning

This repository is dedicated to the pruning of large language models (LLMs). It aims to serve as a comprehensive resource for researchers and practitioners interested in the efficient reduction of model size while maintaining or enhancing performance.

# Contents
- Papers
  - Survey
  - Layer Pruning
- Fine-tuning methods
- Models
- Datasets 
- Tools
- Others
- Citation


## Papers


- [1] Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods
  Arxiv, [Paper](https://arxiv.org/pdf/2402.02834), [Code](https://github.com/Nota-NetsPresso/shortened-llm)
  - Label: Structure Pruning, Depth Pruning
  - Summary: 
- [2] ShortGPT:Layers in Large Language Models are More Redundant Than You Expect
  Arxiv, [Paper](https://arxiv.org/pdf/2403.03853)
  - Label: Structure Pruning, Depth Pruning
  - Summary: ShortGPT propose a metric called Block Influence (BI) as an effective indicator of layer importance. Based on the BI metric, they propose a simple yet effective pruning strategy by removing layers with low BI scores.
- [3] FoldGPT: Simple and Effective Large Language Model Compression Scheme
  Arxiv, [Paper](https://arxiv.org/pdf/2407.00928)
  - Label:
  - Summary: 
- [4] Pruning via Merging: Compressing LLMs via Manifold Alignment Based Layer Merging
  Arxiv, [Paper](https://arxiv.org/pdf/2406.16330)
  - Summary: Structure Pruning, Depth Pruning
- [5] LaCo: Large Language Model Pruning via Layer Collapse
  Arxiv, [Paper](https://arxiv.org/pdf/2402.11187)
  - Label: Structure Pruning, Depth Pruning
  - Summary:
- [6] BlockPruner: Fine-grained Pruning for Large Language Models
  Arxiv, [Paper](https://arxiv.org/pdf/2406.10594)
  - Label:
  - Summary: BlockPruner segments each Transformer layer into MHA and MLP blocks. It then assesses the importance of these blocks using perplexity measures and applies a heuristic search for iterative pruning.
- [7] SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot
  ICML23, [Paper](https://proceedings.mlr.press/v202/frantar23a/frantar23a.pdf)
  - Label:
  - Summary: 
- [8]  LLM-Pruner: On the Structural Pruning of Large Language Models
  Arxiv, [Paper](https://arxiv.org/pdf/2305.11627)
  - Label:
  - Summary: LLM-Pruner details a pruning algorithm that evaluates neuron importance within each layer. Based on the pruning algorithm，It removes neurons with minimal contribution to optimize model efficiency.
- [9] The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction
  Arxiv, [Paper](https://arxiv.org/pdf/2312.13558)
  - Label: Other
  - Summary: This paper selectively removing higher-order components found by SVD of the weight matrices.

  
## Fine-tuning methods
| Name     | Paper         |
|----------|--------------|
| Low-RankAdaptation (LoRA)    | [1]   | 
| ContinuedPretraining (CPT)    | [1]   | 
| CPT+LoRA    | [1]   | 
 

## Models
| Name     | Paper         |
|----------|--------------|
| LLaMA2-7B    | [2],[3],[4],[6],[8]  | 
| LLaMaA2-13B      | [2],[4],[6]    |
| LLaMaA3-8B      | [4]     |
| LLaMaA3-70B      | [4]     |
| Baichuan2-7B   |   [2],[6]           |
| Baichuan2-13B   |   [2],[6]           |
| LLaMA-7B    | [1]    | 
| Vicuna-7b-v1.3    | [1],[8]    | 
| Vicuna-13b-v1.3    | [1]    | 
| Gemma-2B |  [3] |
| TinyLLaMA-1.1B |  [3] |
| Mixtral-7B      | [4]     |
| Qwen1.5-7B      | [6]     |
| Qwen1.5-14B      | [6]     |
| ChatGLM-6B     | [8]   |

## Datasets 
### Calibration Dataset
| Name     | Paper         |
|----------|--------------|
| BookCorpus   | [1]    | 

### For Finetuning
| Name     | Paper         |
|----------|--------------|
| the cleaned version of Alpaca  |   [1],[3],  |
| SlimPajama      |   [1]   |

### For Evaluation
| Name     | Paper         |
|----------|--------------|
| BoolQ   |  [1],[2],[3],[4],[5],[8]  | 
| PIQA    |    [1],[2],[3],[4],[5],[6],[8]|
| WikiText2   |   [1],[3],[6],[8]  |
| PTB  |   [1],[3],[8]  |
| HellaSwag |    [1],[2],[3],[4],[5],[6],[8] |
| WinoGrande |   [1],[3],[6],[8]   |
| ARC-easy |   [1],[6],[8]  |
| ARC-challenge |   [1],[6],[8]   |
| OpenbookQA |   [1],[8]   |
| MMLU |   [2],[3],[4],[5]  |
| CMMLU |   [2]   |
| CMNLI |   [2],[5]   |
| CHID |   [2],[5]   |
| CoQA |   [2],[5]   |
| Race |   [2],[4],[5]   |
| XSum |   [2],[5]   |
| C3 |   [2],[5]   |
| PG19 |   [2]   |
| SCIQ |   [3]   |
| WSC |   [5]   |

## Tools
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

[OpenCompass](https://github.com/open-compass/opencompass)


## 
- Applicability with Quantization

## Others
If the statistics are wrong, please don't hesitate to contact us.

## Citation


## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=yaolu-zjut/Awesome-LLM-Pruning&type=Date)](https://star-history.com/#yaolu-zjut/Awesome-LLM-Pruning&Date)






