# Efficient Inference and Small Language Models: A Survey

## Abstract
Small language models (SLMs) and efficient inference are best understood as a joint design problem rather than a parameter-count contest. Model-side choices—architecture, training, and compression—can reduce capacity, memory footprint, or computation, while serving-side choices such as KV-cache management improve utilization and throughput. These objectives are related but not interchangeable: quantization can save resources without making inference faster, and a serving technique measured on large GPU-hosted models need not transfer to small or on-device deployments. This survey synthesizes retrieved survey-level, primary-paper, documentation, and technical-report evidence. It emphasizes distinct metrics, conditional speedup claims, evaluation methodology, and evidence limits.

## 1. Scope and framing
There is no operationally universal cutoff that makes a model “small.” The retrieved *A Survey of Small Language Models* frames SLMs by retaining useful capability under constraints such as inference hardware, data availability, bandwidth, and generation time; it notes that “small” depends on context and time [1]. Consequently, a parameter count alone is a weak basis for comparison. A model may be small relative to a cloud foundation model but too expensive for a constrained device, or large for an embedded application but economical in a batched server.

The scope here is inference efficiency: how to obtain acceptable task quality at a given latency, throughput, memory, and energy budget. The available retrieved notes do not substantiate a comprehensive catalog of named SLM families or a unified benchmark comparing distillation, pruning, quantization, and serving optimizations. The survey therefore treats method categories as a taxonomy and avoids inventing cross-method rankings.

## 2. What “efficient” means
At least four outcomes should be kept distinct:

- **Quality:** task or benchmark performance at a specified prompt, generation, and evaluation protocol.
- **Latency:** time to first token and inter-token latency, preferably with tail percentiles as well as averages.
- **Throughput:** completed tokens or requests per unit time at a stated concurrency and quality-of-service target.
- **Resource use:** peak accelerator/host memory and, where measured, energy or cost.

The SLM survey specifically distinguishes inference time/throughput from peak memory/footprint [1]. These measures can move in different directions. A smaller weight file does not establish lower end-to-end latency; peak memory reduction can instead enable larger batches or a deployment on smaller hardware. The retrieved quantization survey summary reports that quantization can reduce resource requirements while sometimes slowing inference [2]. Any claim of “efficient” should therefore state the target metric and workload.

## 3. Model-side approaches
The retrieved SLM survey organizes optimization at the architecture, training, and compression levels [1]. It names lightweight architectures and efficient attention under architecture, and pruning, quantization, and knowledge distillation under compression. This is a useful map of design space, not evidence that each technique has the same effect or that one dominates across tasks.

### 3.1 Architecture and training
Architectural efficiency seeks to reduce the computation and memory required to produce useful outputs; training choices seek a model that retains capability within a constrained deployment envelope. The retrieved material supports this broad classification but does not provide a controlled, common benchmark for architectural variants or training recipes. Accordingly, no universal architecture or training recipe is recommended here.

### 3.2 Compression and quantization
Compression changes the representation or capacity of a model. The survey taxonomy includes pruning, quantization, and distillation [1]. Quantization is particularly relevant to inference because lower-precision weights can reduce storage and memory traffic, but a smaller representation does not guarantee faster execution. Kernel support, hardware, implementation, and workload determine whether the representation translates into speed.

AWQ (Activation-aware Weight Quantization) is a concrete low-bit, weight-only method. Its paper describes using activation statistics to identify salient channels and per-channel scaling to reduce quantization error, without backpropagation or reconstruction [3]. The authors report more than 3× speedup over a Hugging Face FP16 implementation on both desktop and mobile GPUs using TinyChat [3]. This is evidence for a particular method and software/hardware comparison—not a universal quantization multiplier. The exact checked abstract establishes “more than 3×”; narrower 3.2–3.3× figures found in secondary notes were not confirmed by the citation check and are not repeated as a precise claim.

Distillation and pruning appear in the SLM survey’s taxonomy [1], but the retrieved sources provide no directly comparable quality/latency results for these approaches. In particular, the evidence here does not establish that a distilled model will outperform quantization, or vice versa, at a fixed quality and deployment budget.

## 4. Serving-system approaches
Model-side compression is not the only lever. Autoregressive serving retains key/value (KV) state across generated tokens, making cache allocation and memory use important to system throughput. PagedAttention applies virtual-memory-style paging to KV cache, arranging per-request cache in non-contiguous blocks; the vLLM system also supports cache sharing [4].

The PagedAttention paper abstract reports 2–4× higher throughput at the same latency than FasterTransformer and Orca in its evaluated popular LLM workloads. It notes that gains vary with sequence lengths, model sizes, and decoding complexity [4]. This is a meaningful primary-paper result, but it does not establish the same gains for SLMs, different hardware, or on-device inference. The vLLM technical report describes up to 24× throughput over Transformers and up to 3.5× over TGI in specific LLaMA/GPU experiments; these are workload-, model-, implementation-, and baseline-specific comparisons [5]. Those larger figures should not be conflated with the paper’s 2–4× abstract result.

Speculative decoding is another serving technique, intended to reduce inter-token latency by using draft predictions that can be verified by a target model. The retrieved vLLM documentation says its usefulness depends on model family, traffic, hardware, and sampling settings, and identifies medium-to-low-QPS, memory-bound settings as a potential use case [6]. The retrieved systematic study warns that measured speedups can fall short of theoretical bounds and that batch-size-one or prototype results may not represent production serving; verification costs and acceptance behavior affect realized speed [7]. Together, these sources argue for measurement under actual traffic rather than extrapolation from a theoretical acceptance rate.

## 5. Interactions and tradeoffs
The approaches target different bottlenecks. Quantization principally changes model weight representation and resource requirements; PagedAttention addresses KV-cache allocation and serving utilization; speculative decoding targets token-generation latency. They can potentially be combined, but their effects are not simply additive. A retrieved 2025 indexed summary reports that tree-style draft verification can become computationally expensive on 4-bit models and describes a hierarchical alternative, while providing insufficient experimental detail for a general conclusion [8]. This example reinforces that system interactions must be measured rather than assumed.

For a latency-sensitive, low-concurrency deployment, a method that reduces inter-token latency may matter more than maximum aggregate throughput. For a multi-request server constrained by KV memory, cache management may increase useful concurrency. For a device constrained by memory capacity, quantization may enable deployment even if speed gains are absent. These are engineering implications of the distinct bottlenecks; the retrieved evidence does not provide a shared experiment validating a single decision rule across these settings.

## 6. Evaluation protocol and reporting
A credible comparison should make its conditions reproducible and should measure the intended end-to-end outcome:

1. **Fix the task and quality target.** Report model/checkpoint, prompt and decoding settings, and quality measures. Compare methods at matched quality, not only matched model names.
2. **Describe the hardware and software path.** Include accelerator/CPU, memory, precision, kernels, runtime and serving engine. AWQ’s reported speedup illustrates why the implementation and baseline matter [3].
3. **Specify workload.** Give input/output length distributions, concurrency, request rate, batch policy, and context lengths. PagedAttention’s reported gains vary across workload characteristics [4].
4. **Report several metrics.** Include time to first token, inter-token latency, end-to-end latency percentiles, throughput at a stated latency/SLO, peak memory, and energy/cost where available. Do not infer speed from model size or memory savings alone [1][2].
5. **Use the production regime.** Measure realistic batch sizes and request patterns. Documentation and systematic study both caution that speculative decoding’s gains depend on traffic and hardware and can diverge from theoretical bounds [6][7].
6. **State the baseline and uncertainty.** Report versioned baselines, repetitions and variability, and distinguish an author-reported maximum from a typical or independently reproduced result.

This protocol follows directly from the retrieved evidence’s repeated qualification of numerical results by model, hardware, workload, implementation, and baseline [2][3][4][5][6][7].

## 7. Evidence limitations
The evidence set contains a survey of SLMs, primary or technical sources on PagedAttention and AWQ, operational documentation, and indexed summaries of additional surveys and studies. It does not contain a unified benchmark that compares architecture changes, distillation, pruning, quantization, cache management, and speculative decoding on the same models and hardware. Some sources are summaries rather than full experimental records [2][8]. Accordingly, reported speedups are cited as source-specific findings and not pooled or ranked. The retrieved materials also do not substantiate broad claims about energy savings or the superiority of particular named SLM families.

## 8. Conclusion
Efficient inference with SLMs is a constrained optimization problem with no single sufficient metric. Model compression can make models fit or reduce resource requirements, but does not by itself prove faster generation. Serving optimizations can raise throughput or reduce latency under particular conditions, but results depend on workload and system configuration. The most defensible practice is to select a method for a diagnosed bottleneck, preserve quality as an explicit constraint, and evaluate end-to-end on the target hardware and traffic. The literature retrieved here gives promising method-specific results, not a universal speedup or a one-size-fits-all ranking.

## References
[1] A Survey of Small Language Models. arxiv. https://arxiv.org/abs/2410.20011 (n.d.)
[2] A Comprehensive Evaluation of Quantization Strategies for Large Language Models. hf-search. https://huggingface.co/papers/2402.16775 (n.d.)
[3] AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. web. https://arxiv.org/html/2306.00978v2 (n.d.)
[4] Efficient Memory Management for Large Language Model Serving with PagedAttention. web. https://dl.acm.org/doi/10.1145/3600006.3613165 (n.d.)
[5] vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention. web. https://vllm.ai/blog/2023-06-20-vllm (n.d.)
[6] Speculative Decoding - vLLM. web. https://docs.vllm.ai/en/stable/features/speculative_decoding/#speculative-decoding (n.d.)
[7] Speculative Decoding: Performance or Illusion?. web. https://arxiv.org/html/2601.11580 (n.d.)
[8] Speculative Decoding Meets Quantization: Compatibility Evaluation and Hierarchical Framework Design. hf-search. https://huggingface.co/papers/2505.22179 (n.d.)
