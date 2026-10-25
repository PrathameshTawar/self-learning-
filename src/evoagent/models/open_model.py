"""
OpenAI-compatible inference backend for Ollama, vLLM, and external endpoints.
Prefers high-reliability SEARCH/REPLACE blocks suited for small LLMs (e.g. Qwen2.5-Coder 1.5B/7B).
"""

import json
import re
from typing import Optional
import httpx
from evoagent.models.base import IModelBackend, ModelPrompt, ModelResponse
from evoagent.tools.patch_tools import SEARCH_REPLACE_BLOCK_RE


class OpenAICompatibleModel(IModelBackend):
    """
    Connects to local or remote OpenAI-compatible endpoints:
    - Ollama (e.g., http://localhost:11434/v1)
    - vLLM / Local HuggingFace server
    - OpenRouter / LiteLLM proxy
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model_name: str = "qwen2.5-coder:1.5b",
        api_key: str = "not-needed",
        timeout: float = 60.0,
        temperature: float = 0.2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature

    def _build_system_prompt(self, prompt: ModelPrompt) -> str:
        system_prompt = (
            "You are EvoAgent, an autonomous software engineering repair model.\n"
            "Analyze the bug report, failure trace, and code context.\n"
            "You MUST respond with:\n"
            "1. A concise hypothesis explaining the root cause.\n"
            "2. A minimal, exact SEARCH/REPLACE block in the following format:\n"
            "<<<<<<< SEARCH\n"
            "[exact code from target file to replace]\n"
            "=======\n"
            "[new replacement code]\n"
            ">>>>>>> REPLACE\n"
            "Use one block per changed location. Each SEARCH section must copy whole lines exactly "
            "and match only one place in the file.\n"
            "Only change lines necessary to fix the failure."
        )
        # Strategy prompt_style gene (Layer B)
        if prompt.system_instruction:
            system_prompt += f"\n{prompt.system_instruction}"
        return system_prompt

    def _build_user_message(self, prompt: ModelPrompt) -> str:
        lessons_text = "\n".join(f"- {l}" for l in prompt.retrieved_lessons) if prompt.retrieved_lessons else "None"
        retry_feedback = ""
        if prompt.previous_attempt_patch:
            retry_feedback = (
                f"\n=== PREVIOUS FAILED ATTEMPT (Attempt #{prompt.attempt_number - 1}) ===\n"
                f"Patch Attempted:\n{prompt.previous_attempt_patch}\n"
                f"Resulting Error / Output:\n{prompt.previous_attempt_error or 'Test failed'}\n"
                "INSTRUCTION: The previous patch failed to resolve the issue. Analyze why it failed and propose an alternative repair.\n\n"
            )

        return (
            f"Task ID: {prompt.task_id}\n"
            f"Target File: {prompt.target_file}\n"
            f"Issue: {prompt.issue_description}\n"
            f"Failure Stack Trace:\n{prompt.failure_trace}\n\n"
            f"{retry_feedback}"
            f"Relevant Lessons from Past Experience:\n{lessons_text}\n\n"
            f"Current File Content:\n{prompt.file_content}\n\n"
            "Provide your hypothesis and SEARCH/REPLACE block fix."
        )

    def generate_repair_proposal(self, prompt: ModelPrompt) -> ModelResponse:
        """Call OpenAI-compatible chat completions endpoint."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self._build_system_prompt(prompt)},
                {"role": "user", "content": self._build_user_message(prompt)},
            ],
            "temperature": prompt.temperature if prompt.temperature is not None else self.temperature,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(url, headers=headers, json=payload)
                res.raise_for_status()
                data = res.json()

                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

                # Extract SEARCH/REPLACE or unified diff
                diff = self._extract_patch(content)
                hypothesis = self._extract_hypothesis(content)

                return ModelResponse(
                    hypothesis=hypothesis,
                    proposed_diff=diff,
                    explanation=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    model_name=self.model_name,
                )

        except Exception as exc:
            return ModelResponse(
                hypothesis=f"Inference error: {str(exc)}",
                proposed_diff="",
                explanation=f"Failed to connect to model endpoint at {self.base_url}: {str(exc)}",
                model_name=self.model_name,
            )

    def _extract_patch(self, text: str) -> str:
        """Extract SEARCH/REPLACE block or diff block from response."""
        # 1. Collect every SEARCH/REPLACE block (multi-location fixes need all of them)
        sr_blocks = [m.group(0) for m in SEARCH_REPLACE_BLOCK_RE.finditer(text.replace("\r\n", "\n"))]
        if sr_blocks:
            return "\n".join(sr_blocks)

        # 2. Extract from markdown ```diff or ```codeblock
        diff_match = re.search(r"```diff\s*(.*?)\s*```", text, re.DOTALL)
        if diff_match:
            return diff_match.group(1).strip()

        generic_match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL)
        if generic_match and ("---" in generic_match.group(1) or "<<<<<<<" in generic_match.group(1)):
            return generic_match.group(1).strip()

        return ""

    def _extract_hypothesis(self, text: str) -> str:
        """Extract the explanation before the first code fence or SEARCH marker."""
        cut = min((i for i in (text.find("```"), text.find("<<<<<<<")) if i >= 0), default=len(text))
        before_code = text[:cut].strip()
        return before_code if before_code else "Proposed repair generated by model."
