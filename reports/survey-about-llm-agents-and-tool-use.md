# Survey of LLM Agents and Tool Use

## Abstract
Tool use extends language-model generation into action: an agent interprets a request, selects capabilities, constructs calls, observes results, and continues or responds. The literature spans tool-learning methods, agent evaluation, and benchmark construction. A useful synthesis separates the action loop into planning, tool discovery and selection, call construction, execution and feedback, and response synthesis. Benchmarks broaden evaluation beyond call formatting, but their task sets, tool catalogs, scoring, and environmental interaction differ. Results are evidence about specified tasks, not universal rankings or proof of dependable deployment.

## Scope and evidence
This survey synthesizes three independent researcher note collections covering two surveys, benchmark papers, and project documentation. The evidence is heterogeneous: some evidence is abstract-level, some evaluation claims derive from paper indexes or extracted text, and benchmark documentation establishes design more clearly than external validity. We distinguish authors' reported results from independently established conclusions. Citations are given as numbered references.

## 1. What tool use entails
Tool use is not simply producing a well-formed function-call object. A working system maps a goal to an executable sequence, selects a capability, supplies valid arguments, interprets outputs, and decides whether to continue, recover, or answer. The tool-learning survey organizes the process into task planning, tool selection, tool calling, and response generation [1]. Errors propagate across stages: a mistaken plan can yield plausible but irrelevant calls; correct selection with malformed arguments can fail; and accurate execution can still be misreported.

This framing suggests evaluating both local competence and end-to-end outcome. Local measures can isolate selection, argument correctness, or call syntax; task completion tests whether interaction achieves the requested result. Neither alone suffices: an outcome score may hide brittleness, while component scores may fail to predict long-horizon success. The evaluation-survey records treat planning and tool use as capabilities within broader agent assessment, alongside memory and self-reflection [2].

## 2. Benchmarks as complementary lenses
API-Bank provides a bounded runnable setting. Its abstract reports 73 API tools and 314 tool-use dialogues containing 753 API calls to assess planning, retrieval, and calling. The authors also report a training collection of 1,888 dialogues spanning 2,138 APIs and 1,000 domains, and a greater-than-26-point tool-utilization advantage for Lynx over Alpaca [3]. That is an author-reported result tied to the study's setup; the retrieved abstract does not establish a comparable protocol across benchmarks.

ToolLLM/ToolBench shifts emphasis toward catalog scale and multi-tool scenarios. Its abstract reports 16,464 real-world REST APIs across 49 categories, generated instructions and solution paths, and ToolEval measures including pass rate under limited budgets and win rate for solution-path usefulness and quality [4]. The authors describe a fine-tuned ToolLLaMA with an API retriever and claim complex-task execution and generalization to unseen APIs. These are reported results, not independent confirmation. API-Bank and ToolBench thus illuminate different designs—bounded runnable evaluation versus broad API coverage—and their headline figures cannot be compared directly [3][4].

UltraTool describes six dimensions spanning planning, tool creation (awareness and creation), and tool usage (awareness, selection, and argument/input usage). Its authors argue that many prior benchmarks emphasize invocation, whereas this framework also tests planning and whether tools need to be created [5]. This broadens the question from “can the agent call a known tool?” to whether it recognizes a capability gap and can create a suitable tool. The retrieved evidence supports the taxonomy, not a general claim that tool creation improves deployment performance.

The m&ms project documents 4K+ multi-step multimodal tasks using 33 tools, including models, public APIs, and image-processing modules. It describes automatically generated plans, human-verified subsets, execution feedback, and evaluation against plans and results [6]. This foregrounds composition across modalities and steps, complementing API-centered benchmarks. Documentation establishes benchmark design, not that tasks fully represent open-world use or measured results transfer to production.

## 3. Evaluation dimensions
A rigorous evaluation should state its unit of success. Dimensions supported by the cited taxonomies and benchmark designs include:

- Planning and decomposition: whether the sequence is feasible and responsive to the goal [1][3].
- Discovery and selection: whether a relevant tool is retrieved or selected from the inventory [3][4].
- Call construction: whether arguments and inputs meet tool requirements [1][5].
- Execution and interaction: whether outputs, feedback, and multi-step dependencies are handled [6][4].
- Outcome and response: whether the task completes within constraints and the final answer reflects evidence [4].
- Broader qualities: reliability, safety, robustness, efficiency/cost, and scalability. Evaluation-survey summaries identify these as concerns and gaps, but do not quantify their prevalence or severity [2][7].

Metrics must align to constructs. API-Bank reports scale and comparative tool-utilization results; ToolEval describes budget-limited pass rate and win rate [3][4]. These are not interchangeable. Reports should disclose tool availability, task distribution, model and tool versions, interaction limits, failure handling, scoring rubric, and output determinism. Human verification can strengthen reference quality; automated judging can increase throughput; neither removes the need to validate judge quality. The retrieved materials lack common protocol detail sufficient to normalize scores.

## 4. Reliability and safety
Tool agents have an expanded failure surface: inappropriate actions, invalid or costly calls, misread outputs, or continued action under uncertainty. Safety assessment should therefore cover authorization and consequences as well as textual content. Evaluation surveys frame safety, robustness, reliability, efficiency, fine-grained assessment, and scalable evaluation as open concerns [2][7]. The evidence supports treating these as priorities, not a conclusion that any benchmark has solved them.

Long-horizon interaction and dynamic environments complicate evaluation. Static suites support repeatability but may not test changing APIs, partial failures, permissions, or compounding errors. The evaluation-survey record calls for realistic and challenging evaluation and highlights scalability; the ACM survey summary emphasizes dynamic/long-horizon interaction and enterprise concerns such as role-based data access and compliance [2][7]. Since these points derive from indexed or abstract-level summaries, they are agenda-setting reports rather than a full examination of the surveys' evidence.

## 5. Synthesis and agenda
Three conclusions follow. First, tool use is best treated as a pipeline or interaction loop, rather than a single call-generation skill [1][5]. Second, benchmarks are plural: API-Bank, ToolBench, UltraTool, and m&ms emphasize bounded API interaction, broad catalog and multi-step paths, utilization stages including tool creation, and multi-step multimodal tasks respectively [6][5][3][4]. Third, scores are conditional on tasks, tools, constraints, and metrics; these studies do not license global rankings or strong claims of real-world reliability [3][4][2].

Useful next steps include reporting component and end-to-end outcomes together; testing recovery from tool errors and changing interfaces; measuring safety, permission compliance, latency, and cost alongside completion; and using held-out tools and distributions to probe generalization. These recommendations are motivated by benchmark diversity and documented evaluation gaps, not proven solutions. Strong claims about practical effectiveness require deployment-relevant evidence, transparent protocols, and replication beyond the abstracts and benchmark descriptions available here.

## References
[1] Tool Learning with Large Language Models: A Survey. arxiv. https://arxiv.org/abs/2405.17935 (n.d.)
[2] Survey on Evaluation of LLM-based Agents. hf-search. https://huggingface.co/papers/2503.16416 (n.d.)
[3] API-Bank. web. https://aclanthology.org/2023.emnlp-main.187/ (n.d.)
[4] ToolLLM. arxiv. https://arxiv.org/abs/2307.16789 (n.d.)
[5] ToolUsage / UltraTool. web. https://aclanthology.org/2024.findings-acl.259.pdf (n.d.)
[6] m&ms benchmark project. web. https://github.com/RAIVNLab/mms (n.d.)
[7] Evaluation and Benchmarking of LLM Agents. web. https://dl.acm.org/doi/abs/10.1145/3711896.3736570 (n.d.)
