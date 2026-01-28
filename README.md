# elicit

DSPy experiments for prompt elicitation and long-context processing.

## Setup

```bash
uv sync
cp .env.example .env  # add OPENAI_API_KEY
```

## Scripts

### elice.py

Iterative prompt elicitation inspired by [dspy-redteam](https://github.com/haizelabs/dspy-redteam). Given an objective, it generates and refines prompts through multiple layers:

1. **Elicit**: Generate a prompt to achieve the objective
2. **Generate**: Get a response from the target LM
3. **Refine**: Critique the prompt based on the response
4. Repeat for N layers

Uses MIPROv2 for optimization. Models: gpt-3.5-turbo (target), gpt-4-0125-preview (agent/judge).

```bash
uv run python elice.py
```

### rlm.py

Process arbitrarily long files using DSPy's [RLM (Recursive Language Model)](https://arxiv.org/abs/2512.24601). RLM stores context as a variable instead of in the prompt, allowing the model to recursively explore and partition content without context degradation.

```bash
uv run python rlm.py <file_path> "<request>"
```

Example:
```bash
uv run python rlm.py chat_log.txt "What are the main topics discussed?"
```

Uses gpt-5-mini by default.
