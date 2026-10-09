# Reinforcement learning for language-model reasoning: evidence and open questions

## Executive summary

The retrieved notes describe several complementary strands rather than a single, settled recipe: executable feedback for code, process-level supervision for mathematical steps, and outcome-verifiable rewards optimized with policy learning. In the notes, DeepSeekMath is credited with introducing GRPO as a critic-free approach for mathematical reasoning, while Tulu 3 is said to have popularized the broader term reinforcement learning with verifiable rewards (RLVR).[1][2][3] The notes also emphasize that these methods inherit practical challenges from RLHF—reward quality, policy optimization stability, and constraints on policy updates—without making reasoning-specific comparisons from the cited PPO overview.[4]

The central evidence caveat is capability expansion. A critical empirical paper summarized in the notes reports improved low-k pass@k but lower large-k coverage for tested RLVR models relative to their base models, interpreting the pattern as more efficient sampling with narrower coverage. A reasoning survey, however, records contrasting results under different training conditions and calls the question unresolved.[3][5] Thus pass@1 improvement alone is not evidence that training has expanded the model's underlying reasoning repertoire.

## Training signals and methods

A useful distinction is the granularity and reliability of feedback. The notes place CodeRL among early approaches using unit tests and execution feedback as objective signals. They describe Math-Shepherd as using a learned process reward model with step-by-step PPO, and DeepSeekMath as the source of critic-free GRPO for math reasoning.[1][6][7] These examples illustrate different design points: executable outcome checks, intermediate-step evaluation, and an optimization method that avoids a separate critic. The notes explicitly warn that the foundational chapter is a secondary overview and does not establish detailed quantitative results for those underlying papers.[1][6][7]

RLVR is a broader label for training against verifiable outcomes, with direct correctness checks distinguished from preference comparisons. The notes say verifiers can target math, code, proofs, or partially scored tasks.[2] The reasoning survey similarly characterizes math and code as comparatively amenable to verifiable rewards, while noting that dependable, fast automated feedback is difficult for open-ended tasks. It also flags that process feedback may help credit assignment but brings reward-model, reward-hacking, and cost concerns.[3]

These reward choices sit within a broader optimization context. The PPO overview in the notes describes conventional RLHF as supervised fine-tuning, reward-model training, then PPO policy optimization, with KL regularization used to constrain updates; it highlights reward-model quality and PPO stability as practical considerations.[4] This is context, not proof that the conventional RLHF pipeline is necessary or superior for reasoning: the source is not reasoning-specific.[4]

## What the evidence supports—and does not

The notes support treating reward construction, data, algorithms, compute, and infrastructure as coupled scaling issues, rather than assuming that more optimization alone suffices.[3] A concise record of a reward-design paper says popular reward models do not always improve RL training, and that refinement techniques intended to prevent reward hacking can make them useful. Because the retrieved record is brief and lacks methods and quantitative detail, it supports caution but not a prescription for a particular refinement.[8]

On generalization and “new” reasoning ability, the evidence is mixed. The critical study reports that its tested RLVR models improved low-k pass@k while base models outperformed them at large k; its authors interpret this as sampling efficiency improving while coverage narrows, and relate trained paths to those already represented in base-model sampling.[5] This is a bounded empirical result, not a theorem that RL cannot create capabilities. The survey notes contrary evidence from ProRL under different conditions and identifies exploration, training duration, and regularization as relevant factors in the disagreement.[3] Accordingly, evaluations should report more than a single pass@1 figure and specify model, benchmark, sampling budget, and training conditions.[3][5]

## Practical implications and limits

1. **Match reward to the task.** Favor executable or otherwise checkable outcomes where correctness can be verified; for open-ended tasks, the notes do not establish a universally reliable automated reward.[2][3]
2. **Make reward granularity explicit.** Outcome reward and step-level process reward address different feedback needs; process signals may help credit assignment but introduce their own model-quality, hacking, and cost risks.[3][6]
3. **Measure coverage as well as efficiency.** Compare low- and high-k performance against the base model, since the reported capability debate turns on sampling breadth as well as success at modest sample counts.[3][5]
4. **Treat optimization and infrastructure as part of the method.** The notes identify PPO stability and KL constraints in the RLHF context and cite compute, data, algorithms, and infrastructure as broader scaling challenges.[4][3]

The available evidence is uneven: some foundational claims come through a secondary overview, one reward-design source is only a concise search record, and the capability debate is summarized through a survey and one empirical paper. The notes do not justify precise cross-method performance rankings or a universal claim that RLVR either expands or merely concentrates reasoning ability.[1][3][5][8]

## References
[1] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. web. https://arxiv.org/abs/2402.03300 (2024-02-05)
[2] Tulu 3: Pushing Frontiers in Open Language Model Post-Training. web. https://arxiv.org/abs/2411.15124 (2024-11-22)
[3] A Survey of Reinforcement Learning for Large Reasoning Models. arxiv. https://arxiv.org/abs/2509.08827 (2025-10-09)
[4] Secrets of RLHF in Large Language Models Part I: PPO. web. https://arxiv.org/abs/2307.04964 (2023-07-10)
[5] Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?. web. https://arxiv.org/html/2504.13837 (2025-11-24)
[6] Math-Shepherd: Verify and Reinforce LLMs Step-by-Step Without Human Annotations. web. https://arxiv.org/abs/2312.08935 (2023-12-15)
[7] CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning. web. https://arxiv.org/abs/2207.01780 (2022-07-05)
[8] On Designing Effective RL Reward at Training Time for LLM Reasoning. hf-search. https://huggingface.co/papers/2410.15115 (2024-10-19)
