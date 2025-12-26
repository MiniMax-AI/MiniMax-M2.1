<div align="center">
  <picture>
    <source srcset="figures/MiniMaxLogo-Dark.png" media="(prefers-color-scheme: dark)">
      <img src="figures/MiniMaxLogo-Light.png" width="60%" alt="MiniMax">
    </source>
  </picture>
</div>
<hr>

<div align="center" style="line-height: 1.4; font-size:16px; margin-top: 30px;">
  Join Our 
  <a href="https://platform.minimaxi.com/docs/faq/contact-us" target="_blank" style="font-size:17px; margin: 2px;">
    💬 WeChat
  </a> | 
  <a href="https://discord.com/invite/hvvt8hAye6" target="_blank" style="font-size:17px; margin: 2px;">
    🧩 Discord
  </a> 
  community.
</div>
<div align="center" style="line-height: 1.2; font-size:16px;">
  <a href="https://agent.minimax.io/" target="_blank" style="display: inline-block; margin: 4px;">
    MiniMax Agent
  </a> | 
  <a href="https://platform.minimax.io/docs/guides/text-generation" target="_blank" style="display: inline-block; margin: 4px;">
    ⚡️ API
  </a> | 
  <a href="https://github.com/MiniMax-AI/MiniMax-MCP" style="display: inline-block; margin: 4px;">
    MCP
  </a> |
  <a href="https://www.minimax.io" target="_blank" style="display: inline-block; margin: 4px;">
    MiniMax Website
  </a> 
</div>
<div align="center" style="line-height: 1.2; font-size:16px; margin-bottom: 30px;">
  <a href="https://huggingface.co/MiniMaxAI" target="_blank" style="margin: 2px;">
    🤗 Hugging Face 
  </a> | 
  <a href="https://github.com/MiniMax-AI/MiniMax-M2.1" target="_blank" style="margin: 2px;">
    🐙 GitHub
  </a> | 
  <a href="https://www.modelscope.cn/organization/MiniMax" target="_blank" style="margin: 2px;">
    🤖️ ModelScope
  </a> | 
  <a href="https://github.com/MiniMax-AI/MiniMax-M2.1/blob/main/LICENSE" style="margin: 2px;">
    📄 License: Modified-MIT
  </a>
</div>

# Meet MiniMax-M2.1

Today, we are handing **MiniMax-M2.1** over to the open-source community. This release is more than just a parameter update; it is a significant step toward democratizing top-tier agentic capabilities.

M2.1 was built to shatter the stereotype that high-performance agents must remain behind closed doors. We have optimized the model specifically for robustness in coding, tool use, instruction following, and long-horizon planning. From automating multilingual software development to executing complex, multi-step office workflows, MiniMax-M2.1 empowers developers to build the next generation of autonomous applications—all while being fully transparent, controllable, and accessible.

We believe true intelligence should be within reach. M2.1 is our commitment to the future, and a powerful new tool in your hands.

<p align="center">
  <img width="100%" src="figures/bench.png">
</p>

## How to Use

- The MiniMax-M2.1 API is now live on the MiniMax Open Platform: https://platform.minimax.io/docs/guides/text-generation
- Our product MiniMax Agent, built on MiniMax-M2.1, is now publicly available: https://agent.minimax.io/
- The MiniMax-M2.1 model weights are now open-source, allowing for local deployment and use: https://huggingface.co/MiniMaxAI/MiniMax-M2.1

## Benchmarks

MiniMax-M2.1 delivers a significant leap over M2 on core software engineering leaderboards. It shines particularly bright in multilingual scenarios, where it outperforms Claude Sonnet 4.5 and closely approaches Claude Opus 4.5.

| Benchmark | MiniMax-M2.1 | MiniMax-M2 | Claude Sonnet 4.5 | Claude Opus 4.5 | Gemini 3 Pro | GPT-5.2 (thinking) | DeepSeek V3.2 |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| SWE-bench Verified | 74.0 | 69.4 | 77.2 | 80.9 | 78.0 | 80.0 | 73.1 |
| Multi-SWE-bench | 49.4 | 36.2 | 44.3 | 50.0 | 42.7 | x | 37.4 |
| SWE-bench Multilingual | 72.5 | 56.5 | 68 | 77.5 | 65.0 | 72.0 | 70.2 |
| Terminal-bench 2.0 | 47.9 | 30.0 | 50.0 | 57.8 | 54.2 | 54.0 | 46.4 |

We also evaluated MiniMax-M2.1 on SWE-bench Verified across a variety of coding agent frameworks. The results highlight the model's exceptional framework generalization and robust stability.

Furthermore, across specific benchmarks—including test case generation, code performance optimization, code review, and instruction following—MiniMax-M2.1 demonstrates comprehensive improvements over M2. In these specialized domains, it consistently matches or exceeds the performance of Claude Sonnet 4.5.

| Benchmark | MiniMax-M2.1 | MiniMax-M2 | Claude Sonnet 4.5 | Claude Opus 4.5 | Gemini 3 Pro | GPT-5.2 (thinking) | DeepSeek V3.2 |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| SWE-bench Verified (Droid) | 71.3 | 68.1 | 72.3 | 75.2 | x | x | 67.0 |
| SWE-bench Verified (mini-swe-agent) | 67.0 | 61.0 | 70.6 | 74.4 | 71.8 | 74.2 | 60.0 |
| SWT-bench | 69.3 | 32.8 | 69.5 | 80.2 | 79.7 | 80.7 | 62.0 |
| SWE-Perf | 3.1 | 1.4 | 3.0 | 4.7 | 6.5 | 3.6 | 0.9 |
| SWE-Review | 8.9 | 3.4 | 10.5 | 16.2 | x | x | 6.4 |
| OctoCodingbench | 26.1 | 13.3 | 22.8 | 36.2 | 22.9 | x | 26.0 |

To evaluate the model's full-stack capability to architect complete, functional applications "from zero to one," we established a novel benchmark: [VIBE (Visual & Interactive Benchmark for Execution in Application Development)](https://huggingface.co/datasets/MiniMaxAI/VIBE). This suite encompasses five core subsets: Web, Simulation, Android, iOS, and Backend. Distinguishing itself from traditional benchmarks, VIBE leverages an innovative Agent-as-a-Verifier (AaaV) paradigm to automatically assess the interactive logic and visual aesthetics of generated applications within a real runtime environment.

MiniMax-M2.1 delivers outstanding performance on the VIBE aggregate benchmark, achieving an average score of 88.6—demonstrating robust full-stack development capabilities. It excels particularly in the VIBE-Web (91.5) and VIBE-Android (89.7) subsets.

| Benchmark | MiniMax-M2.1 | MiniMax-M2 | Claude Sonnet 4.5 | Claude Opus 4.5 | Gemini 3 Pro |
| ----- | ----- | ----- | ----- | ----- | ----- |
| VIBE (Average) | 88.6 | 67.5 | 85.2 | 90.7 | 82.4 |
| VIBE-Web | 91.5 | 80.4 | 87.3 | 89.1 | 89.5 |
| VIBE-Simulation | 87.1 | 77.0 | 79.1 | 84.0 | 89.2 |
| VIBE-Android | 89.7 | 69.2 | 87.5 | 92.2 | 78.7 |
| VIBE-iOS | 88.0 | 39.5 | 81.2 | 90.0 | 75.8 |
| VIBE-Backend | 86.7 | 67.8 | 90.8 | 98.0 | 78.7 |

MiniMax-M2.1 also demonstrates steady improvements over M2 in both long-horizon tool use and comprehensive intelligence metrics.

| Benchmark | MiniMax-M2.1 | MiniMax-M2 | Claude Sonnet 4.5 | Claude Opus 4.5 | Gemini 3 Pro | GPT-5.2 (thinking) | DeepSeek V3.2 |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| Toolathlon | 43.5 | 16.7 | 38.9 | 43.5 | 36.4 | 41.7 | 35.2 |
| BrowseComp | 47.4 | 44.0 | 19.6 | 37.0 | 37.8 | 65.8 | 51.4 |
| BrowseComp (context management) | 62.0 | 56.9 | 26.1 | 57.8 | 59.2 | 70.0 | 67.6 |
| AIME25 | 83.0 | 78.0 | 88.0 | 91.0 | 96.0 | 98.0 | 92.0 |
| MMLU-Pro | 88.0 | 82.0 | 88.0 | 90.0 | 90.0 | 87.0 | 86.0 |
| GPQA-D | 83.0 | 78.0 | 83.0 | 87.0 | 91.0 | 90.0 | 84.0 |
| HLE w/o tools | 22.2 | 12.5 | 17.3 | 28.4 | 37.2 | 31.4 | 22.2 |
| LCB | 81.0 | 83.0 | 71.0 | 87.0 | 92.0 | 89.0 | 86.0 |
| SciCode | 41.0 | 36.0 | 45.0 | 50.0 | 56.0 | 52.0 | 39.0 |
| IFBench | 70.0 | 72.0 | 57.0 | 58.0 | 70.0 | 75.0 | 61.0 |
| AA-LCR | 62.0 | 61.0 | 66.0 | 74.0 | 71.0 | 73.0 | 65.0 |
| 𝜏²-Bench Telecom | 87.0 | 87.0 | 78.0 | 90.0 | 87.0 | 85.0 | 91.0 |

> **Evaluation Methodology Notes**:
> - **SWE-bench Verified**: Tested on internal infrastructure using [Claude Code](https://github.com/anthropics/claude-code), [Droid](https://factory.ai/), or [mini-swe-agent](https://github.com/SWE-agent/mini-SWE-agent) as scaffolding. By default, we utilized Claude Code metrics. When using Claude Code, the default system prompt was overridden. Results represent the average of 4 runs.
> - **Multi-SWE-Bench & SWE-bench Multilingual & SWT-bench & SWE-Perf**: Tested on internal infrastructure using Claude Code as scaffolding, with the default system prompt overridden. Results represent the average of 4 runs.
> - **Terminal-bench 2.0**: Tested using Claude Code on our internal evaluation framework. We verified the full dataset and fixed environmental issues. Timeout limits were removed, while all other configurations remained consistent with official settings. Results represent the average of 4 runs.
> - **SWE Review**: Built upon the SWE framework, this internal benchmark for code defect review covers diverse languages and scenarios, evaluating both defect recall and hallucination rates. A review is deemed "correct" only if the model accurately identifies the target defect and ensures all other reported findings are valid and free of hallucinations. All evaluations are executed using Claude Code, with final results reflecting the average of four independent runs per test case. We plan to open-source this benchmark soon.
> - **OctoCodingbench**: An internal benchmark focused on long-horizon instruction following for Code Agents in complex development scenarios. It conducts end-to-end behavioral supervision within a dynamic environment spanning diverse tech stacks and scaffolding frameworks. The core objective is to evaluate the model's ability to integrate and execute "composite instruction constraints"—encompassing System Prompts (SP), User Queries, Memory, Tool Schemas, and specifications such as `Agents.md`, `Claude.md`, and `Skill.md`. Adopting a strict "single-violation-failure" scoring mechanism, the final result is the average pass rate across 4 runs, quantifying the model's robustness in translating static constraints into precise behaviors. We plan to open-source this benchmark soon.
> - **VIBE**: An internal benchmark that utilizes Claude Code as scaffolding to automatically verify a program's interactive logic and visual effects. Scores are calculated through a unified pipeline comprising requirement sets, containerized deployment, and dynamic interaction environments. Final results represent the average of 3 runs. We have open-sourced this benchmark at [VIBE](https://huggingface.co/datasets/MiniMaxAI/VIBE).
> - **Toolathlon**: The evaluation protocol remains consistent with the original paper.
> - **BrowseComp**: All scores were obtained using the same agent framework as [WebExplorer](https://arxiv.org/pdf/2509.06501) (Liu et al. 2025), with only minor fine-tuning of tool descriptions. We utilized the same 103-sample GAIA text-only validation subset as WebExplorer.
> - **BrowseComp (context management)**: When token usage exceeds 30% of the maximum context window, we retain the first AI response, the last five AI responses, and the tool outputs, discarding the remaining content.
> - **AIME25 ~ 𝜏²-Bench Telecom**: Derived from internal testing based on the evaluation datasets and methodology referenced in the [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/).

## Local Deployment Guide

Download the model from HuggingFace repository: https://huggingface.co/MiniMaxAI/MiniMax-M2.1

We recommend using the following inference frameworks (listed alphabetically) to serve the model:

### SGLang

We recommend using [SGLang](https://docs.sglang.io/) to serve MiniMax-M2.1. Please refer to our [SGLang Deployment Guide](./docs/sglang_deploy_guide.md).

### vLLM

We recommend using [vLLM](https://github.com/vllm-project/vllm) to serve MiniMax-M2.1. Please refer to our [vLLM Deployment Guide](./docs/vllm_deploy_guide.md).

### Transformers

We recommend using [Transformers](https://github.com/huggingface/transformers) to serve MiniMax-M2.1. Please refer to our [Transformers Deployment Guide](./docs/transformers_deploy_guide.md).

### Other Inference Engines

- [KTransformers](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/MiniMax-M2.1-Tutorial.md)

### Inference Parameters

We recommend using the following parameters for best performance: `temperature=1.0`, `top_p = 0.95`, `top_k = 40`. Default system prompt:

```
You are a helpful assistant. Your name is MiniMax-M2.1 and is built by MiniMax.
```

## Tool Calling Guide

Please refer to our [Tool Calling Guide](./docs/tool_calling_guide.md).

## Contact Us

Contact us at [model@minimax.io](mailto:model@minimax.io).
