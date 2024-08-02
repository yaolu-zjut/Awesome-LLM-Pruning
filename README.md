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
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2402.02834), [Code](https://github.com/Nota-NetsPresso/shortened-llm)
  - Label: Structure Pruning, Depth Pruning
  - Summary: 
- [2] ShortGPT:Layers in Large Language Models are More Redundant Than You Expect
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2403.03853)
  - Label: Structure Pruning, Depth Pruning
  - Summary: ShortGPT propose a metric called Block Influence (BI) as an effective indicator of layer importance. Based on the BI metric, they propose a simple yet effective pruning strategy by removing layers with low BI scores.
- [3] FoldGPT: Simple and Effective Large Language Model Compression Scheme
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2407.00928)
  - Label:
  - Summary: 
- [4] Pruning via Merging: Compressing LLMs via Manifold Alignment Based Layer Merging
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2406.16330)
  - Label:
  - Summary: Structure Pruning, Depth Pruning
- [5] LaCo: Large Language Model Pruning via Layer Collapse
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2402.11187)
  - Label: Structure Pruning, Depth Pruning
  - Summary:
- [6] BlockPruner: Fine-grained Pruning for Large Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2406.10594), [Code](https://github.com/MrGGLS/BlockPruner)
  - Label: Structure Pruning, 
  - Summary: BlockPruner segments each Transformer layer into MHA and MLP blocks. It then assesses the importance of these blocks using perplexity measures and applies a heuristic search for iterative pruning.
- [7] SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot
  - Publication: ICML23, [Paper](https://proceedings.mlr.press/v202/frantar23a/frantar23a.pdf)
  - Label:
  - Summary: 
- [8] LLM-Pruner: On the Structural Pruning of Large Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2305.11627)
  - Label:
  - Summary: LLM-Pruner details a pruning algorithm that evaluates neuron importance within each layer. Based on the pruning algorithm，It removes neurons with minimal contribution to optimize model efficiency.
- [9] The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2312.13558), [Code](https://github.com/pratyushasharma/laser)
  - Label: Other, Low-Rank Decomposition
  - Summary: This paper selectively removing higher-order components found by SVD of the weight matrices.
- [10] Plug-and-Play: An Efficient Post-Training Pruning Method for Large Language Models
  - Publication: ICLR 2024, [Paper](https://openreview.net/pdf?id=Tr0lPx9woF), [Code](https://github.com/biomedical-cybernetics/Relative-importance-and-activation-pruning)
  - Label:
  - Summary: 
- [11] Wanda: A Simple and Effective Pruning Approach For Large Language Models
  - Publication: ICLR 2024, [Paper](https://arxiv.org/pdf/2306.11695.pdf), [Code](https://github.com/locuslab/wanda)
  - Label:
  - Summary:
- [12] LoRAP: Transformer Sub-Layers Deserve Differentiated Structured Compression for Large Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2404.09695)
  - Label: Other, Low-Rank Decomposition
  - Summary:
- [13] Structured Pruning for Large Language Models Using Coupled Components Elimination and Minor Fine-tuning
  - Publication: NAACL 2024 findings, [Paper](https://aclanthology.org/2024.findings-naacl.1.pdf)
  - Label:  
  - Summary:
- [14] LoRAShear: Efficient Large Language Model Structured Pruning and Knowledge Recovery
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2310.18356)
  - Label: 
  - Summary:
- [15] Pruning Large Language Models via Accuracy Predictor
  - Publication: ICASSP 2024, [Paper](https://arxiv.org/pdf/2309.09507)
  - Label: 
  - Summary:
- [16] SliceGPT: Compress Large Language Models by Deleting Rows and Columns
  - Publication:  ICLR 2024, [Paper](https://arxiv.org/pdf/2401.15024)
  - Label: 
  - Summary:
- [17] APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference
  - Publication: ICML 2024 Oral, [Paper](https://arxiv.org/pdf/2401.12200), [Code](https://github.com/ROIM1998/APT)
  - Label: 
  - Summary:
- [18] How to Prune Your Language Model: Recovering Accuracy on the “Sparsity May Cry” Benchmark
  - Publication: Conference on Parsimony and Learning 2024, [Paper](https://proceedings.mlr.press/v234/kurtic24a/kurtic24a.pdf)
  - Label: 
  - Summary:
- [19] The LLM Surgeon
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2312.17244)
  - Label: 
  - Summary:
- [20] The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models
  - Publication: EMNLP 2022, [Paper](https://arxiv.org/pdf/2203.07259), [Code](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT)
  - Label: 
  - Summary:
- [21] Pushing Gradient towards Zero: A Novel Pruning Method for Large Language Models
  - Publication: Arxiv, [Paper](https://openreview.net/pdf?id=IU4L7wiwxw)
  - Label: 
  - Summary:
- [22] SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2402.09025)
  - Label: 
  - Summary:
- [23] Beyond Size: How Gradients Shape Pruning Decisions in Large Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2311.04902), [Code](https://github.com/VILA-Lab/GBLM-Pruner)
  - Label: 
  - Summary:
- [24] Compressing LLMs: The Truth is Rarely Pure and Never Simple
  - Publication: ICLR 2024, [Paper](https://arxiv.org/pdf/2310.01382), [Code](https://github.com/VITA-Group/llm-kick)
  - Label: 
  - Summary:
- [25] E-Sparse: Boosting the Large Language Model Inference through Entropy-based N:M Sparsity 
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2310.15929)
  - Label: 
  - Summary: 
- [26] The Emergence of Essential Sparsity in Large Pre-trained Models: The Weights that Matter 
  - Publication: NeurIPS 2023, [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/7a69ab48efcbb0153e72d458fb091969-Paper-Conference.pdf)
  - Label: 
  - Summary: 
- [27] ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models
  - Publication: ICLR 2024, [Paper](https://arxiv.org/pdf/2310.02998), [Code](https://github.com/ylsung/ECoFLaP)
  - Label: 
  - Summary:
- [28] Pruning via Ranking (PvR): A unified structured pruning approach
  - Publication: Arxiv, [Paper](https://openreview.net/pdf?id=rO62BY3dYc)
  - Label: 
  - Summary:
- [29] The Need for Speed: Pruning Transformers with One Recipe
  - Publication: ICLR 2024, [Paper](https://arxiv.org/pdf/2403.17921), [Code](https://github.com/Skhaki18/optin-transformer-pruning)
  - Label: 
  - Summary:
- [30] Differentiable Model Scaling using Differentiable Topk
  - Publication: ICML 2024, [Paper](https://arxiv.org/pdf/2405.07194)
  - Label: 
  - Summary:
- [31] Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2402.05406)
  - Label: 
  - Summary:
- [32] Flextron: Many-in-One Flexible Large Language Model
  - Publication: ICML 2024, [Paper](https://arxiv.org/pdf/2406.10260)
  - Label: 
  - Summary:
- [33] Compresso: Structured Pruning with Collaborative Prompting Learns Compact Large Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2310.05015)
  - Label: 
  - Summary:
- [34] Structural Pruning of Large Language Models via Neural Architecture Search
  - Publication: Arxiv, [Paper](https://assets.amazon.science/40/08/3b42096c4427a35fee1ea612401d/structural-pruning-of-large-language-models-via-neural-architecture-search.pdf)
  - Label: 
  - Summary:
- [35] Fluctuation-Based Adaptive Structured Pruning for Large Language Models
  - Publication: AAAI 2024, [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28960), [Code](https://github.com/CASIA-IVA-Lab/FLAP)
  - Label: 
  - Summary: 
- [36] The Unreasonable Ineffectiveness of the Deeper Layers
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2403.17887)
  - Label: Structure Pruning, Layer Pruning
  - Summary:
- [37] Junk DNA Hypothesis: Pruning Small Pre-Trained Weights Irreversibly and Monotonically Impairs “Difficult" Downstream Tasks in LLMs 
  - Publication: ICML 2024, [Paper](https://openreview.net/pdf?id=EfUrTeuUfy), [Code](https://github.com/VITA-Group/Junk_DNA_Hypothesis)
  - Label: 
  - Summary:
- [38] Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning
  - Publication: ICLR 2024, [Paper](https://arxiv.org/pdf/2310.06694)
  - Label: 
  - Summary:
- [39] BESA: Pruning Large Language Models with Blockwise Parameter-Efficient Sparsity Allocation Outlier 
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2402.16880)
  - Label: 
  - Summary:
- [40] Pruner-Zero: Evolving Symbolic Pruning Metric from scratch for Large Language Models 
  - Publication: ICML 2024, [Paper](https://arxiv.org/pdf/2406.02924)
  - Label: 
  - Summary:
- [41] Pruning as a Domain-specific LLM Extractor
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2405.06275)
  - Label: 
  - Summary:
- [42] MINI-LLM: Memory-Efficient Structured Pruning for Large Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2407.11681)
  - Label: 
  - Summary:
- [43] Optimization-based Structural Pruning for Large Language Models without Back-Propagation
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2406.10576)
  - Label: 
  - Summary:
- [44] Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2310.05175), [Code](https://github.com/luuyin/OWL)
  - Label: 
  - Summary:
- [45] ALPS: Improved Optimization for Highly Sparse One-Shot Pruning for Large Language Models  
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2406.07831)
  - Label: 
  - Summary:
- [46] Nuteprune: Efficient progressive pruning with numerous teachers for large language models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2402.09773)
  - Label: 
  - Summary:
- [47] MoreauPruner: Robust Pruning of Large Language Models against Weight Perturbations 
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2406.07017), [Code](https://github.com/ShiningSord/MoreauPruner)
  - Label: 
  - Summary:
- [48] Large Language Model Pruning 
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2406.00030)
  - Label: 
  - Summary:
- [49] ShadowLLM: Predictor-based Contextual Sparsity for Large Language Models 
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2406.16635), [Code](https://github.com/abdelfattah-lab/shadow_llm/)
  - Label: 
  - Summary:
- [50] Turbo Sparse: Achieving LLM SOTA Performance with Minimal Activated Parameters
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2406.05955)
  - Label: 
  - Summary:
- [51] One-Shot Sensitivity-Aware Mixed Sparsity Pruning for Large Language Models
  - Publication: ICASSP 2024, [Paper](https://ieeexplore.ieee.org/abstract/document/10445737)
  - Label: 
  - Summary:
- [52] Sparsity May Cry: Let Us Fail (Current) Sparse Neural Networks Together!
  - Publication: ICLR 2023, [Paper](https://arxiv.org/pdf/2303.02141)
  - Label: 
  - Summary:
- [53] Compressing large language models by streamlining the unimportant layer
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2403.19135)
  - Label: Structure Pruning, Layer Pruning
  - Summary:
- [54] FinerCut: Finer-grained Interpretable Layer Pruning for Large Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2405.18218)
  - Label: 
  - Summary:
- [55] ZipLM: Inference-Aware Structured Pruning of Language Models
  - Publication: NeurIPS 2023, [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/ced46a50befedcb884ccf0cbe8c3ad23-Paper-Conference.pdf)
  - Label: 
  - Summary:
- [56] Achieving Sparse Activation in Small Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2406.06562)
  - Label: 
  - Summary:
- [57] Greedy Output Approximation: Towards Efficient Structured Pruning for LLMs Without Retraining
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2407.19126)
  - Label: 
  - Summary:
- [58] Efficient Pruning of Large Language Model with Adaptive Estimation Fusion
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2403.10799)
  - Label: 
  - Summary:
- [59] A deeper look at depth pruning of LLMs
  - Publication: ICML 2024 Workshop, [Paper](https://arxiv.org/pdf/2407.16286), [Code](https://github.com/shoaibahmed/llm_depth_pruning)
  - Label: Structure Pruning, Layer Pruning
  - Summary:
- [60] 
  - Publication: Arxiv, [Paper]()
  - Label: 
  - Summary: 
- [61] 
  - Publication: Arxiv, [Paper]()
  - Label: 
  - Summary:
- [62] 
  - Publication: Arxiv, [Paper]()
  - Label: 
  - Summary:
- [63] 
  - Publication: Arxiv, [Paper]()
  - Label: 
  - Summary:
- [64] 
  - Publication: Arxiv, [Paper]()
  - Label: 
  - Summary:
- [65] 
  - Publication: Arxiv, [Paper]()
  - Label: 
  - Summary:


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
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [OpenCompass](https://github.com/open-compass/opencompass)


## 
- Applicability with Quantization

## Others
If the statistics are wrong, please don't hesitate to contact us.

## Citation


## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=yaolu-zjut/Awesome-LLM-Pruning&type=Date)](https://star-history.com/#yaolu-zjut/Awesome-LLM-Pruning&Date)






