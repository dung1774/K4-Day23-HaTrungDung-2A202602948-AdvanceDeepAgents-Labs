# World Models: A Survey of Concepts, Methods, Evidence, and Open Challenges

## Abstract

A world model is an internal predictive representation of an environment that enables an agent to anticipate consequences of actions. In model-based reinforcement learning (MBRL), it is usually an explicit or learned transition/reward model used for planning or policy learning; in embodied AI, the term also covers latent-state predictors, action-conditioned video generators, and systems that jointly predict states and actions. The promise is to replace some costly real interaction with simulated experience and foresight. This survey synthesizes foundational and recent work across these meanings. It reviews the conceptual lineage, model architectures and uses, evaluation, safety, and limitations. The central conclusion is that predictive usefulness, not visual plausibility alone, is the decisive criterion: generated futures may look coherent yet fail to capture task-relevant physics or support reliable action. Results remain bounded by benchmarks, horizons, and domains, so world models should be treated as fallible decision aids rather than validated replicas of reality.

## 1. Scope and terminology

“World model” is an umbrella rather than a single agreed architecture. Broad definitions include systems that represent the present, infer mechanisms, or predict future dynamics; the decision-oriented use simulates action consequences to guide choices.[1] MBRL has a narrower operational meaning: an agent uses a known or learned model of an MDP in conjunction with planning or policy/value learning.[2] This distinction matters. A video predictor that extrapolates observations is not automatically a control model, and an MBRL system need not render pixels.

A useful abstraction is a latent state z_t inferred from observations, a transition predictor p(z_{t+1}|z_t,a_t), and optionally a reward or observation predictor. The model may be deterministic or probabilistic, explicit or latent, and trained from passive or action-labelled experience. Partial observability may require a recurrent or belief state. The practical test is whether predictions preserve the variables relevant to decisions over the intended horizon—not whether every detail of the environment is reproduced.

## 2. Why build one?

Real-world interaction can be expensive, slow, or hazardous. A learned model permits simulated rollouts for planning and policy optimization, potentially improving sample efficiency and reducing the need for risky trials.[2][3] The same foresight can support counterfactual comparison, policy evaluation, data generation, and visual planning. A safety-focused review distinguishes the agent’s downstream outcomes (such as reward or violations) from generated-output metrics (such as image/video similarity or geometric errors); the two answer different questions.[4]

This promise has a fundamental caveat: policy learning exploits model errors. The model’s prediction discrepancy can make a policy optimized in simulation perform poorly in the real environment.[3] Model accuracy is therefore not an abstract forecasting goal; its required fidelity depends on actions, states, and outcomes the planner will encounter.

## 3. Historical and methodological development

### 3.1 Latent simulation and learned dynamics

A classic demonstration by Ha and Schmidhuber trained a recurrent generative model to compress spatiotemporal experience, then trained a compact controller in the generated environment and transferred it to the task environment.[5] In CarRacing-v0, the paper reports a mean score of 906 ± 21 over 100 trials for the full system. This result is an existence proof for latent “dreams,” not evidence of general transfer or safety; the benchmark is a specific simulated game.

Later MBRL work has organized design around learning the model, deciding how to use it, and training the policy while controlling the consequences of model error.[3] Models can be used for online lookahead, synthetic rollout generation, or latent-space policy optimization. Dreamer exemplifies the latter: it learns compact predictive states from images and trains behavior through imagined sequences. A Google Research account reports evaluation on 20 continuous-control tasks, an average score of 823 versus 786 for D4PG, and 20 times fewer environment interactions; these are claims for that stated benchmark, not universal comparative guarantees.[6]

### 3.2 Video and embodied world models

In robotics, action-conditioned video models predict observations under specified actions, while other systems condition on task instructions or explicit spatial representations. Such predictions can act as context for downstream action models, or furnish virtual scenarios for planning and evaluation. Recent robotics survey work groups uses into action/data generation, dynamics and reward modeling, policy evaluation, and visual planning.[7] Joint world-action approaches further seek to connect predictive state modelling directly to action generation, contrasting this with purely reactive observation-to-action policies.[8]

The representation and conditioning are consequential: a model may condition on actions, embodiment, language, or spatial structure. But a rendered future is only a proposal about what could happen. It must remain grounded in actual observations and control-relevant variables; successful image synthesis alone does not establish accurate dynamics.

## 4. Evaluation: what evidence is needed?

A rigorous evaluation should separate at least four levels:

1. **Prediction:** Does the model predict relevant state variables, events, and uncertainty at one-step and multi-step horizons?
2. **Decision utility:** Does planning or policy learning with the model improve held-out task performance over suitable baselines, under comparable interaction and compute budgets?
3. **Transfer and robustness:** Does performance persist under new initial conditions, disturbances, embodiments, and distribution shifts, rather than only the training benchmark?
4. **Safety:** Are failures, rare events, constraint violations, and uncertainty-calibration errors measured in the intended deployment setting?

Perceptual measures such as video similarity or frame-level error can be informative but cannot substitute for decision and safety outcomes. The safety review catalogues output metrics including FID/FVD, PSNR, distance errors, collisions, and traffic-rule violations, while emphasizing the gap between model-output evaluation and downstream agent performance.[4] Likewise, a virtual evaluation that correlates with a real-world outcome on a limited set of tasks is encouraging but does not validate all tasks or environments.[9]

Uncertainty also needs careful treatment. The MBRL survey distinguishes irreducible aleatoric variability from epistemic uncertainty due to limited data and notes that prediction reliability affects planning.[2] A robust agent must know when its model is out of distribution and either gather evidence, defer, or use a conservative fallback; a single confidence score is not itself a safety guarantee.

## 5. Limitations and failure modes

**Compounding error.** Small transition errors can accumulate across imagined trajectories, degrading long-horizon plans. The MBRL literature identifies multi-step prediction and model generalization as persistent challenges.[2][3]

**Plausibility versus physical validity.** Video can be temporally convincing while violating physical constraints or misrepresenting contacts. Robotics survey evidence identifies hallucinated or inconsistent predictions, physics violations, instruction-following failures, and high data/training/inference costs.[7]

**Action grounding.** A model may not follow the requested intervention precisely; downstream action selection may ignore the interaction region or attend to irrelevant image features. Recent analyses report this mismatch and propose alignment techniques, while noting that mitigation is partial and evaluations cover limited tasks.[10]

**Horizon, latency, and coverage.** Short predictive horizons, long-horizon drift, expensive generation, limited scenario diversity, and weak embodiment transfer constrain deployment. These are especially consequential for real-time robotics, where slow or stale predictions can make otherwise accurate foresight unusable.[7][11]

**Benchmark validity.** Strong scores on a set of control tasks or a simulated game establish performance only within the reported protocol. They do not establish safe transfer to rare events, new sensors, or real-world conditions.[4][5][6]

## 6. Research priorities

First, evaluation should connect predictive errors to downstream decisions and safety failures, with long-horizon and distribution-shift tests rather than relying on output fidelity alone.[4][7] Second, models should expose calibrated epistemic uncertainty and support conservative planning, fallback behaviour, and human oversight when evidence is weak.[2][4] Third, action grounding requires systematic tests of interventions, contacts, and embodiment changes; promising spatial conditioning and feature-alignment methods need broader independent validation.[9][10] Fourth, efficiency must be measured end-to-end, including inference latency and the cost of data acquisition, not merely training-time scaling.[7][11] Finally, common task suites should distinguish simulator performance, virtual evaluation, and physical deployment, and report failure distributions alongside averages.

## 7. Conclusion

World models provide a general route to foresight: compress experience into predictive structure and use it to simulate possible outcomes before acting. Their potential spans MBRL, robotics, and other embodied domains, but the term covers materially different systems. The strongest evidence supports benefits in defined tasks and benchmarks, not a general claim that learned models are reliable substitutes for the world. The field’s key challenge is to turn plausible prediction into calibrated, action-grounded, safe decision support—and to validate that capability under the conditions where it will actually be used.

## References
[1] Understanding World or Predicting Future? A Comprehensive Survey of World Models. web. https://dl.acm.org/doi/10.1145/3746449 (2025-09-09)
[2] Model-based Reinforcement Learning: A Survey. web. https://liacs.leidenuniv.nl/~plaata1/papers/model_based_rl_survey_fnt.pdf (2018)
[3] A Survey on Model-Based Reinforcement Learning. web. https://arxiv.org/pdf/2206.09328 (n.d.)
[4] World Models: The Safety Perspective. web. https://arxiv.org/html/2411.07690v1 (2024-11)
[5] Recurrent World Models Facilitate Policy Evolution. arxiv. https://arxiv.org/abs/1809.01999 (2018-09-04)
[6] Introducing Dreamer: Scalable Reinforcement Learning Using World Models. web. https://research.google/blog/introducing-dreamer-scalable-reinforcement-learning-using-world-models/ (2020-03-18)
[7] Robotic Video World Models: A Survey of Applications and Challenges. web. https://arxiv.org/html/2601.07823v2 (2026-09-17)
[8] World Action Models: The Next Frontier in Embodied AI. hf-search. https://huggingface.co/papers/2605.12090 (2026-05-12)
[9] OSCAR: Omni-Embodiment Action-Conditioned World Model for Robotics. web. https://arxiv.org/abs/2606.04463 (2026-06-03)
[10] Making Foresight Actionable: Repurposing Representation Alignment in World Action Models. web. https://arxiv.org/abs/2606.12217 (2026-06-10)
[11] Say, Dream, and Act: Learning Video World Models for Instruction-Driven Robot Manipulation. web. https://arxiv.org/abs/2602.10717 (n.d.)
