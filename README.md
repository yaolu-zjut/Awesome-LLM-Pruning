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


## Papers


- [1] Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods
  Arxiv, [Paper](https://arxiv.org/pdf/2402.02834), [Code](https://github.com/Nota-NetsPresso/shortened-llm) 
- [2] ShortGPT:Layers in Large Language Models are More Redundant Than You Expect
  Arxiv, [Paper](https://arxiv.org/pdf/2403.03853)
- [3] FoldGPT: Simple and Effective Large Language Model Compression Scheme
  Arxiv, [Paper](https://arxiv.org/pdf/2407.00928)
- [4] Pruning via Merging: Compressing LLMs via Manifold Alignment Based Layer Merging
  Arxiv, [Paper](https://arxiv.org/pdf/2406.16330)
- [5] LaCo: Large Language Model Pruning via Layer Collapse
  Arxiv, [Paper](https://arxiv.org/pdf/2402.11187)

  
## Fine-tuning methods
| Name     | Paper         |
|----------|--------------|
| Low-RankAdaptation (LoRA)    | [1]   | 
| ContinuedPretraining (CPT)    | [1]   | 
| CPT+LoRA    | [1]   | 
 

## Models
| Name     | Paper         |
|----------|--------------|
| LLaMA2-7B    | [2],[3],[4]  | 
| LLaMaA2-13B      | [2],[4]    |
| LLaMaA3-8B      | [4]     |
| LLaMaA3-70B      | [4]     |
| Baichuan2-7B   |   [2]           |
| Baichuan2-13B   |   [2]           |
| LLaMA-7B    | [1]    | 
| Vicuna-7b-v1.3    | [1]    | 
| Vicuna-13b-v1.3    | [1]    | 
| Gemma-2B |  [3] |
| TinyLLaMA-1.1B |  [3] |
| Mixtral-7B      | [4]     |

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
| BoolQ   |  [1],[2],[3],[4]   | 
| PIQA    |    [1],[2],[3],[4], |
| WikiText2   |   [1],[3],  |
| PTB  |   [1],[3],  |
| HellaSwag |    [1],[2],[3],[4]  |
| WinoGrande |   [1],[3],   |
| ARC-easy |   [1]   |
| ARC-challenge |   [1]   |
| OpenbookQA |   [1]   |
| MMLU |   [2],[3],[4]  |
| CMMLU |   [2]   |
| CMNLI |   [2]   |
| CHID |   [2]   |
| CoQA |   [2]   |
| Race |   [2],[4]   |
| XSum |   [2]   |
| C3 |   [2]   |
| PG19 |   [2]   |
| SCIQ |   [3]   |

## Tools
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
[OpenCompass](https://github.com/open-compass/opencompass)


## 
- Applicability with Quantization






