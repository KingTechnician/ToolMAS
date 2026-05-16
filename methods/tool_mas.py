import re
from typing import Dict, List, Optional, Tuple

import argparse
import torch

from models import ModelWrapper, _past_length
from prompts import build_tool_mas_message
from utils import (
    run_python_with_stdout,
    extract_gsm8k_answer,
    normalize_answer,
    extract_markdown_python_block,
    run_with_timeout,
)

_MAX_TOOL_RESULT_CHARS = 2000

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\w+)\((.*?)\)\s*</tool_call>",
    re.DOTALL,
)


class ToolMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 4,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        tool_max_iters: int = 5,
        tool_timeout: int = 10,
        tool_latent_steps: int = -1,
        args: argparse.Namespace = None,
    ) -> None:
        self.model = model
        self.latent_steps = latent_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tool_max_iters = tool_max_iters
        self.tool_timeout = tool_timeout
        # Allow separate latent step count for tool result digestion; fall back to latent_steps.
        self.tool_latent_steps = tool_latent_steps if tool_latent_steps >= 0 else latent_steps
        self.task = args.task if args else "gsm8k"

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _parse_tool_call(self, text: str) -> Optional[Tuple[str, str]]:
        matches = _TOOL_CALL_RE.findall(text)
        if not matches:
            return None
        name, args = matches[-1]
        return name.strip(), args.strip()

    def _execute_tool(self, name: str, args: str) -> str:
        if name == "python_exec":
            result = run_python_with_stdout(args, self.tool_timeout)
            return result[:_MAX_TOOL_RESULT_CHARS]
        return f"Error: unknown tool '{name}'"

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) != 1:
            raise ValueError(
                "ToolMASMethod requires generate_bs=1 (batch size 1); "
                f"got {len(items)} items."
            )
        item = items[0]
        return [self._run_single(item)]

    def _run_single(self, item: Dict) -> Dict:
        question = item["question"]
        messages = build_tool_mas_message(question, self.task)

        prompts, input_ids, attention_mask, _ = self.model.prepare_chat_batch(
            [messages], add_generation_prompt=True
        )

        past_kv: Optional[Tuple] = None
        trace: List[Dict] = []
        final_text = ""

        for iteration in range(self.tool_max_iters):
            generated_list, past_kv = self.model.generate_text_batch(
                input_ids,
                attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                past_key_values=past_kv,
            )
            generated = generated_list[0]

            trace.append({
                "name": f"ToolAgent (iter {iteration})",
                "role": "react",
                "input": prompts[0] if iteration == 0 else "<continued from cache>",
                "output": generated,
                "iter": iteration,
            })

            tool_call = self._parse_tool_call(generated)
            if tool_call is None:
                final_text = generated
                break

            tool_name, tool_args = tool_call
            result = self._execute_tool(tool_name, tool_args)

            trace.append({
                "name": f"Tool: {tool_name} (iter {iteration})",
                "role": "tool",
                "input": tool_args,
                "output": result,
                "iter": iteration,
            })

            # Digest tool result into latent KV cache.
            result_text = f"<tool_result>{result}</tool_result>"
            result_enc = self.model.tokenizer(
                result_text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            result_ids = result_enc["input_ids"].to(self.model.device)
            result_mask = result_enc["attention_mask"].to(self.model.device)

            past_kv = self.model.generate_latent_batch(
                result_ids,
                attention_mask=result_mask,
                latent_steps=self.tool_latent_steps,
                past_key_values=past_kv,
            )

            # Seed next text generation with a minimal newline token so
            # generate_text_batch has a non-empty input to continue from.
            cont_enc = self.model.tokenizer(
                "\n",
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = cont_enc["input_ids"].to(self.model.device)
            attention_mask = torch.ones_like(input_ids)

        else:
            # Exhausted iterations without a tool-free response — use last generated text.
            final_text = generated if trace else ""

        return self._score(item, final_text, trace)

    # ------------------------------------------------------------------
    # Scoring (mirrors latent_mas.py:201-236)
    # ------------------------------------------------------------------

    def _score(self, item: Dict, final_text: str, trace: List[Dict]) -> Dict:
        task = self.task
        gold = item.get("gold", "")
        solution = item.get("solution", "")

        if task in ("mbppplus", "humanevalplus"):
            pred = extract_markdown_python_block(final_text)
            if pred is None:
                ok = False
                error_msg = "No python code block found"
            else:
                python_code_to_exe = pred + "\n" + gold
                ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
        elif task in ("aime2024", "aime2025"):
            pred = normalize_answer(extract_gsm8k_answer(final_text))
            try:
                ok = int(pred) == int(str(gold).strip())
            except (ValueError, TypeError):
                ok = False
        else:
            pred = normalize_answer(extract_gsm8k_answer(final_text))
            ok = (pred == gold) if (pred and gold) else False

        return {
            "question": item["question"],
            "gold": gold,
            "solution": solution,
            "prediction": pred,
            "raw_prediction": final_text,
            "agents": trace,
            "correct": ok,
        }

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
