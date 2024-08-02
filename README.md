# Awesome-LLM-Pruning

This repository is dedicated to the pruning of large language models (LLMs). It aims to serve as a comprehensive resource for researchers and practitioners interested in the efficient reduction of model size while maintaining or enhancing performance.

# Contents
- Papers
  - Survey
  - Unstructured Pruning
  - Structured Pruning
  - Others
- Fine-tuning methods
- Models
- Datasets 
- Tools
- Others
- Citation


## Papers
### Survey
- [1] Model Compression and Efficient Inference for Large Language Models: A Survey
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2402.09748)
  - Summary: Quantization, Pruning, Knowledge Distillation, Efficient Architecture Design, Dynamic Networks
- [2] A survey on model compression for large language models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2308.07633)
  - Label: Quantization, Pruning, Knowledge Distillation, Low-Rank Factorization 
  - Summary:
- [3] A Comprehensive Survey of Compression Algorithms for Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2401.15347)
  - Label: Pruning, Quantization, Knowledge Distillation, Low-Rank Factorization, Parameter Sharing, Efficient Architecture Design
  - Summary:
- [4] Faster and Lighter LLMs: A Survey on Current Challenges and Way Forward
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2402.01799), [Code](https://github.com/nyunAI/Faster-LLM-Survey)
  - Label: Pruning, Quantization, Knowledge Distillation, Low-Rank Factorization, System Level Approaches
  - Summary:
- [5] A Survey on Efficient Inference for Large Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2404.14294)
  - Label:  Data-level Optimization, Model-level Optimization, System-level Optimization
  - Summary:
- [6] A survey of resource-efficient llm and multimodal foundation models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2401.08092), [Website](https://github.com/UbiquitousLearning/Efficient_Foundation_Model_Survey)
  - Label: Large Language Model, Multimodal Large Language Model, Resource-efficient Architectures,  Resource-efficient Algorithms, Resource-efficient Systems
  - Summary:
- [7] Beyond Efficiency: A Systematic Survey of Resource-Efficient Large Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2401.00625), [Code](https://github.com/tiingweii-shii/Awesome-Resource-Efficient-LLM-Papers)
  - Label: 
  - Summary:
 
  

### Unstructured Pruning
> Unstructured pruning involves zeroing out individual weights, resulting in higher sparsity ratios while maintaining better performance. However, without specialized hardware support, it can be challenging to achieve inference speedup with this method.
- [1] Wanda: A Simple and Effective Pruning Approach For Large Language Models
  - Publication: ICLR 2024, [Paper](https://arxiv.org/pdf/2306.11695.pdf), [Code](https://github.com/locuslab/wanda)
  - Label: Magnitude-based Pruning, N:M sparsity
  - Summary:
- [2] E-Sparse: Boosting the Large Language Model Inference through Entropy-based N:M Sparsity 
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2310.15929)
  - Label: Magnitude-based Pruning, N:M sparsity
  - Summary:
- [3] Plug-and-Play: An Efficient Post-Training Pruning Method for Large Language Models
  - Publication: ICLR 2024, [Paper](https://openreview.net/pdf?id=Tr0lPx9woF), [Code](https://github.com/biomedical-cybernetics/Relative-importance-and-activation-pruning)
  - Label: Magnitude-based Pruning
  - Summary: 
- [4] SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot
  - Publication: ICML23, [Paper](https://proceedings.mlr.press/v202/frantar23a/frantar23a.pdf)
  - Label: Loss-based Pruning, N:M sparsity
  - Summary:
- [5] One-Shot Sensitivity-Aware Mixed Sparsity Pruning for Large Language Models
  - Publication: ICASSP 2024, [Paper](https://ieeexplore.ieee.org/abstract/document/10445737)
  - Label: Loss-based Pruning
  - Summary:
- [6] Beyond Size: How Gradients Shape Pruning Decisions in Large Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2311.04902), [Code](https://github.com/VILA-Lab/GBLM-Pruner)
  - Label: Loss-based Pruning
  - Summary:
- [7] Pushing Gradient towards Zero: A Novel Pruning Method for Large Language Models
  - Publication: Arxiv, [Paper](https://openreview.net/pdf?id=IU4L7wiwxw)
  - Label: Loss-based Pruning
  - Summary:
- [8] The LLM Surgeon
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2312.17244)
  - Label: Loss-based Pruning
  - Summary:
- [9] Dynamic Sparse No Training: Training-Free Fine-tuning for Sparse LLMs
  - Publication: ICLR 2024, [Paper](https://openreview.net/pdf?id=1ndDmZdT4g), [Code](https://github.com/zyxxmu/DSnoT)
  - Label: 
  - Summary: 
- [10] Flash-LLM: Enabling Cost-Effective and Highly-Efficient Large Generative Model Inference with Unstructured Sparsity
  - Publication: Proceedings of the VLDB Endowment, [Paper](https://dl.acm.org/doi/10.14778/3626292.3626303)
  - Label: Hardware Support
  - Summary:
- [11] Prune and tune: Improving efficient pruning techniques for massive language models
  - Publication: ICLR 2023 Tiny Paper, [Paper](https://openreview.net/pdf?id=cKlgcx7nSZ)
  - Label: 
  - Summary:
- [12] The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models
  - Publication: EMNLP 2022, [Paper](https://arxiv.org/pdf/2203.07259), [Code](https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT)
  - Label: 
  - Summary:
- [13] The Emergence of Essential Sparsity in Large Pre-trained Models: The Weights that Matter
  - Publication: Arxiv, [Paper](https://openreview.net/pdf?id=bU9hwbsVcy), [Code](https://github.com/VITA-Group/essential_sparsity)
  - Label: 
  - Summary: 
- [14] PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2312.15230)
  - Label: 
  - Summary:
- [15] Gradient-Free Adaptive Global Pruning for Pre-trained Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2402.17946v1), [Code](https://github.com/BaiTheBest/SparseLLM)
  - Label: 
  - Summary:
- [16] Fast and Effective Weight Update for Pruned Large Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2401.02938), [Code](https://github.com/fmfi-compbio/admm-pruning)
  - Label: 
  - Summary:
- [17] COPAL: Continual Pruning in Large Language Generative Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2405.02347v1)
  - Label: 
  - Summary: 
- [18] Dependency-Aware Semi-Structured Sparsity of GLU Variants in Large Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2405.01943v1)
  - Label: 
  - Summary: 
### Structured Pruning
> Structured pruning achieves inference speedup by removing entire network structures, such as , and . As a result, the sparsity ratios in structured pruned models are typically lower than those in unstructured ones.
- [1] Fluctuation-Based Adaptive Structured Pruning for Large Language Models
  - Publication: AAAI 2024, [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28960), [Code](https://github.com/CASIA-IVA-Lab/FLAP)
  - Label: Magnitude-based Pruning
  - Summary:
- [2] SliceGPT: Compress Large Language Models by Deleting Rows and Columns
  - Publication:  ICLR 2024, [Paper](https://arxiv.org/pdf/2401.15024)
  - Label: Magnitude-based Pruning
  - Summary:
- [3] Mini-GPTs: Efficient Large Language Models through Contextual Pruning
  - Publication:  ICLR 2024, [Paper](https://arxiv.org/pdf/2312.12682), [Code](https://github.com/tval2/contextual-pruning)
  - Label: Magnitude-based Pruning
  - Summary:
- [4] LLM-Pruner: On the Structural Pruning of Large Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2305.11627)
  - Label: Loss-based Pruning
  - Summary: LLM-Pruner details a pruning algorithm that evaluates neuron importance within each layer. Based on the pruning algorithm，It removes neurons with minimal contribution to optimize model efficiency.
- [5] LoRAShear: Efficient Large Language Model Structured Pruning and Knowledge Recovery
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2310.18356)
  - Label: Loss-based Pruning
  - Summary:
- [6] APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference
  - Publication: ICML 2024 Oral, [Paper](https://arxiv.org/pdf/2401.12200), [Code](https://github.com/ROIM1998/APT)
  - Label: Loss-based Pruning
  - Summary:
- [7] Pruning Large Language Models via Accuracy Predictor
  - Publication: ICASSP 2024, [Paper](https://arxiv.org/pdf/2309.09507)
  - Label: Loss-based Pruning
  - Summary:
- [8] Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2402.02834), [Code](https://github.com/Nota-NetsPresso/shortened-llm)
  - Label: Loss-based Pruning, Layer Pruning
  - Summary: 
- [9] Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning
  - Publication: ICLR 2024, [Paper](https://arxiv.org/pdf/2310.06694)
  - Label: Regularization-based Pruning
  - Summary:
- [10] ShortGPT:Layers in Large Language Models are More Redundant Than You Expect
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2403.03853)
  - Label: Structure Pruning, Depth Pruning
  - Summary: ShortGPT propose a metric called Block Influence (BI) as an effective indicator of layer importance. Based on the BI metric, they propose a simple yet effective pruning strategy by removing layers with low BI scores.
- [11] FoldGPT: Simple and Effective Large Language Model Compression Scheme
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2407.00928)
  - Label:
  - Summary: 
- [12] Pruning via Merging: Compressing LLMs via Manifold Alignment Based Layer Merging
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2406.16330)
  - Label:
  - Summary: Structure Pruning, Depth Pruning
- [13] LaCo: Large Language Model Pruning via Layer Collapse
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2402.11187)
  - Label: Structure Pruning, Depth Pruning
  - Summary:
- [14] BlockPruner: Fine-grained Pruning for Large Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2406.10594), [Code](https://github.com/MrGGLS/BlockPruner)
  - Label: Structure Pruning, 
  - Summary: BlockPruner segments each Transformer layer into MHA and MLP blocks. It then assesses the importance of these blocks using perplexity measures and applies a heuristic search for iterative pruning.
- [15] The Truth is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2312.13558), [Code](https://github.com/pratyushasharma/laser)
  - Label: Other, Low-Rank Decomposition
  - Summary: This paper selectively removing higher-order components found by SVD of the weight matrices.
- [16] LoRAP: Transformer Sub-Layers Deserve Differentiated Structured Compression for Large Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2404.09695)
  - Label: Other, Low-Rank Decomposition
  - Summary:
- [17] Structured Pruning for Large Language Models Using Coupled Components Elimination and Minor Fine-tuning
  - Publication: NAACL 2024 findings, [Paper](https://aclanthology.org/2024.findings-naacl.1.pdf)
  - Label:  
  - Summary:
- [18] How to Prune Your Language Model: Recovering Accuracy on the “Sparsity May Cry” Benchmark
  - Publication: Conference on Parsimony and Learning 2024, [Paper](https://proceedings.mlr.press/v234/kurtic24a/kurtic24a.pdf)
  - Label: 
  - Summary:
- [19] SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2402.09025)
  - Label: 
  - Summary:
- [21] The Emergence of Essential Sparsity in Large Pre-trained Models: The Weights that Matter 
  - Publication: NeurIPS 2023, [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/7a69ab48efcbb0153e72d458fb091969-Paper-Conference.pdf)
  - Label: 
  - Summary: 
- [22] ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models
  - Publication: ICLR 2024, [Paper](https://arxiv.org/pdf/2310.02998), [Code](https://github.com/ylsung/ECoFLaP)
  - Label: 
  - Summary:
- [23] Pruning via Ranking (PvR): A unified structured pruning approach
  - Publication: Arxiv, [Paper](https://openreview.net/pdf?id=rO62BY3dYc)
  - Label: 
  - Summary:
- [24] The Need for Speed: Pruning Transformers with One Recipe
  - Publication: ICLR 2024, [Paper](https://arxiv.org/pdf/2403.17921), [Code](https://github.com/Skhaki18/optin-transformer-pruning)
  - Label: 
  - Summary:
- [25] Differentiable Model Scaling using Differentiable Topk
  - Publication: ICML 2024, [Paper](https://arxiv.org/pdf/2405.07194)
  - Label: 
  - Summary:
- [26] Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2402.05406)
  - Label: 
  - Summary:
- [27] Flextron: Many-in-One Flexible Large Language Model
  - Publication: ICML 2024, [Paper](https://arxiv.org/pdf/2406.10260)
  - Label: 
  - Summary:
- [28] Compresso: Structured Pruning with Collaborative Prompting Learns Compact Large Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2310.05015)
  - Label: 
  - Summary:
- [29] Structural Pruning of Large Language Models via Neural Architecture Search
  - Publication: Arxiv, [Paper](https://assets.amazon.science/40/08/3b42096c4427a35fee1ea612401d/structural-pruning-of-large-language-models-via-neural-architecture-search.pdf)
  - Label: 
  - Summary:
- [30] The Unreasonable Ineffectiveness of the Deeper Layers
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2403.17887)
  - Label: Structure Pruning, Layer Pruning
  - Summary:
- [31] Junk DNA Hypothesis: Pruning Small Pre-Trained Weights Irreversibly and Monotonically Impairs “Difficult" Downstream Tasks in LLMs 
  - Publication: ICML 2024, [Paper](https://openreview.net/pdf?id=EfUrTeuUfy), [Code](https://github.com/VITA-Group/Junk_DNA_Hypothesis)
  - Label: 
  - Summary:
- [32] BESA: Pruning Large Language Models with Blockwise Parameter-Efficient Sparsity Allocation Outlier 
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2402.16880)
  - Label: 
  - Summary:
- [33] Pruner-Zero: Evolving Symbolic Pruning Metric from scratch for Large Language Models 
  - Publication: ICML 2024, [Paper](https://arxiv.org/pdf/2406.02924)
  - Label: 
  - Summary:
- [34] Pruning as a Domain-specific LLM Extractor
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2405.06275)
  - Label: 
  - Summary:
- [35] MINI-LLM: Memory-Efficient Structured Pruning for Large Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2407.11681)
  - Label: 
  - Summary:
- [36] Optimization-based Structural Pruning for Large Language Models without Back-Propagation
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2406.10576)
  - Label: 
  - Summary:
- [37] Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2310.05175), [Code](https://github.com/luuyin/OWL)
  - Label: 
  - Summary:
- [38] ALPS: Improved Optimization for Highly Sparse One-Shot Pruning for Large Language Models  
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2406.07831)
  - Label: 
  - Summary:
- [39] Nuteprune: Efficient progressive pruning with numerous teachers for large language models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2402.09773)
  - Label: 
  - Summary:
- [40] MoreauPruner: Robust Pruning of Large Language Models against Weight Perturbations 
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2406.07017), [Code](https://github.com/ShiningSord/MoreauPruner)
  - Label: 
  - Summary:
- [41] Large Language Model Pruning 
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2406.00030)
  - Label: 
  - Summary:
- [42] ShadowLLM: Predictor-based Contextual Sparsity for Large Language Models 
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2406.16635), [Code](https://github.com/abdelfattah-lab/shadow_llm/)
  - Label: 
  - Summary:
- [43] Turbo Sparse: Achieving LLM SOTA Performance with Minimal Activated Parameters
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2406.05955)
  - Label: 
  - Summary:
- [44] Sparsity May Cry: Let Us Fail (Current) Sparse Neural Networks Together!
  - Publication: ICLR 2023, [Paper](https://arxiv.org/pdf/2303.02141)
  - Label: 
  - Summary:
- [45] FinerCut: Finer-grained Interpretable Layer Pruning for Large Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2405.18218)
  - Label: 
  - Summary:
- [46] ZipLM: Inference-Aware Structured Pruning of Language Models
  - Publication: NeurIPS 2023, [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/ced46a50befedcb884ccf0cbe8c3ad23-Paper-Conference.pdf)
  - Label: 
  - Summary:
- [47] Achieving Sparse Activation in Small Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2406.06562)
  - Label: 
  - Summary:
- [48] Greedy Output Approximation: Towards Efficient Structured Pruning for LLMs Without Retraining
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2407.19126)
  - Label: 
  - Summary:
- [49] Efficient Pruning of Large Language Model with Adaptive Estimation Fusion
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2403.10799)
  - Label: 
  - Summary:
- [50] A deeper look at depth pruning of LLMs
  - Publication: ICML 2024 Workshop, [Paper](https://arxiv.org/pdf/2407.16286), [Code](https://github.com/shoaibahmed/llm_depth_pruning)
  - Label: Structure Pruning, Layer Pruning
  - Summary:
- [51] Compact Language Models via Pruning and Knowledge Distillation
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2407.14679)
  - Label: 
  - Summary: 
- [52] BlockLLM: Memory-Efficient Adaptation of LLMs by Selecting and Optimizing the Right Coordinate Blocks
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2406.17296)
  - Label: 
  - Summary:
- [53] Structured Pruning for Efficient Generative Pre-trained Language Models
  - Publication: ACL 2023 Findings, [Paper](https://aclanthology.org/2023.findings-acl.692.pdf)
  - Label: 
  - Summary:
- [54] Reconstruct the Pruned Model without Any Retraining
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2407.13331)
  - Label: 
  - Summary:
- [55] Pruning before Fine-tuning: A Retraining-free Compression Framework for Pre-trained Language Models
  - Publication: LREC-COLING 2024, [Paper](https://aclanthology.org/2024.lrec-main.1162.pdf), [Code](https://github.com/applewpj/P-pruning)
  - Label: 
  - Summary:
- [56] OSSCAR: One-Shot Structured Pruning in Vision and Language Models with Combinatorial Optimization
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2403.12983)
  - Label: 
  - Summary:
- [57] NASH: A Simple Unified Framework of Structured Pruning for Accelerating Encoder-Decoder Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2310.10054), [Code](https://github.com/jongwooko/NASH-Pruning-Official)
  - Label: 
  - Summary:
- [58] LoRAPrune: Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning
  - Publication: ACL 2024 Findings, [Paper](https://arxiv.org/pdf/2305.18403)
  - Label: 
  - Summary:
- [59] Not all Layers of LLMs are Necessary during Inference
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2403.02181)
  - Label: 
  - Summary:
- [60] Streamlining Redundant Layers to Compress Large Language Models
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2403.19135)
  - Label: Structure Pruning, Layer Pruning
  - Summary:
- [61] Compact Language Models via Pruning and Knowledge Distillation
  - Publication: Arxiv, [Paper](https://www.arxiv.org/pdf/2407.14679), [Code](https://github.com/NVlabs/Minitron)
  - Label: 
  - Summary:
### Others
- [1] How Does Calibration Data Affect the Post-training Pruning and Quantization of Large Language Models?
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2311.09755)
  - Label: 
  - Summary:
- [2] Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models: Enhancing Performance and Reducing Inference Costs
  - Publication: Arxiv, [Paper](https://arxiv.org/pdf/2407.00945)
  - Label: 
  - Summary:
- [3] Compressing LLMs: The Truth is Rarely Pure and Never Simple
  - Publication: ICLR 2024, [Paper](https://arxiv.org/pdf/2310.01382), [Code](https://github.com/VITA-Group/llm-kick)
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






