"""
Unit tests for Layer C adapter training: chat formatting, loss masking, length filtering, GPU checks,
and a CPU smoke run of the training loop when torch/transformers/peft are installed.
"""

import json
from pathlib import Path
import pytest
from evoagent.learning.dataset import DatasetCurator, scrub_secrets
from evoagent.learning.finetune import (
    IGNORE_INDEX,
    TrainingDataError,
    _filter_kwargs,
    check_gpu,
    length_summary,
    manifest_mismatch,
    pad_batch,
    rows_for_split,
    tokenize_example,
    tokenize_rows,
)
from evoagent.memory.store import InMemoryMemoryStore

GB = 1024**3


class ChatMLTokenizer:
    """Character-level stand-in for a Qwen tokenizer with the ChatML template."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return text + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def row(completion: str = "fix", split: str = "train", user: str = "bug") -> dict:
    return {
        "task_id": "t",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": user}],
        "prompt": f"sys\n\n{user}",
        "completion": completion,
        "split": split,
    }


def decode(ids) -> str:
    return "".join(chr(i) for i in ids)


def test_only_the_assistant_reply_is_learned():
    feature = tokenize_example(ChatMLTokenizer(), row("return x"), max_seq_length=10_000)

    learned = [tok for tok, label in zip(feature["input_ids"], feature["labels"]) if label != IGNORE_INDEX]
    assert decode(learned) == "return x<|im_end|>\n"
    # The prompt, rendered exactly as at inference, ends with the generation prompt
    prompt = decode(tok for tok, label in zip(feature["input_ids"], feature["labels"]) if label == IGNORE_INDEX)
    assert prompt.endswith("<|im_start|>assistant\n") and "<|im_start|>user\nbug" in prompt


def test_episode_rows_without_messages_become_a_user_turn():
    feature = tokenize_example(ChatMLTokenizer(), {"prompt": "fix it", "completion": "ok"}, 10_000)
    assert decode(feature["input_ids"]).startswith("<|im_start|>user\nfix it")


def test_too_long_examples_are_skipped_not_truncated():
    tokenizer = ChatMLTokenizer()
    full = len(tokenize_example(tokenizer, row(), 10_000)["input_ids"])
    assert tokenize_example(tokenizer, row(), full) is not None
    assert tokenize_example(tokenizer, row(), full - 1) is None

    features, lengths = tokenize_rows(tokenizer, [row(), row(user="x" * 500)], max_seq_length=full)
    assert len(features) == 1 and len(lengths) == 2
    assert length_summary(lengths, full)["fit"] == 1


def test_template_that_does_not_prefix_the_prompt_is_rejected():
    class BrokenTemplate(ChatMLTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return ("PROMPT" if add_generation_prompt else "CONVERSATION") + messages[-1]["content"]

    with pytest.raises(TrainingDataError):
        tokenize_example(BrokenTemplate(), row(), 10_000)


def test_padding_is_masked_out_of_attention_and_loss():
    batch = pad_batch(
        [{"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
         {"input_ids": [4], "attention_mask": [1], "labels": [4]}],
        pad_token_id=0,
    )
    assert batch["input_ids"][1] == [4, 0, 0]
    assert batch["attention_mask"][1] == [1, 0, 0]
    assert batch["labels"][1] == [4, IGNORE_INDEX, IGNORE_INDEX]


def test_val_and_test_rows_never_reach_training():
    rows = [row(split="train"), row(split="val"), row(split="test"), {**row(), "split": None}]
    assert len(rows_for_split(rows, "train")) == 2
    assert len(rows_for_split(rows, "val")) == 1


def test_gtx_1650_passes_the_gpu_check_with_fp16():
    notes = check_gpu("NVIDIA GeForce GTX 1650", 4 * GB, (7, 5), "fp16")
    assert not any(n.startswith("ERROR") for n in notes)
    assert any("SDPA" in n for n in notes)
    assert any(n.startswith("ERROR") for n in check_gpu("NVIDIA GeForce GTX 1650", 4 * GB, (7, 5), "bf16"))
    assert any(n.startswith("ERROR") for n in check_gpu("GT 1030", 2 * GB, (6, 1), "fp16"))
    assert check_gpu("RTX 3060", 12 * GB, (8, 6), "bf16") == []


def test_renamed_training_arguments_are_mapped_and_unknown_ones_reported():
    class TransformersV5Args:
        def __init__(self, output_dir, warmup_steps=0, eval_strategy="no"):
            pass

    messages = []
    kwargs = _filter_kwargs(TransformersV5Args, {"output_dir": "x", "warmup_ratio": 0.03, "made_up": 1}, messages.append)
    # Silently dropping warmup_ratio would train without warmup on transformers 5
    assert kwargs == {"output_dir": "x", "warmup_steps": 0.03}
    assert messages == ["[warning] this transformers version ignores: made_up"]


def test_secrets_are_scrubbed_before_export():
    text = "key sk-abcdefghijklmnopqrstuvwxyz123456 and Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123"
    scrubbed = scrub_secrets(text)
    assert "sk-abc" not in scrubbed and "abcdefghijklmnopqrstuvwxyz0123" not in scrubbed
    assert scrubbed.count("[REDACTED]") == 2 and "Bearer [REDACTED]" in scrubbed
    assert scrub_secrets("def f(x): return x") == "def f(x): return x"


def test_export_dedupes_repeated_trials_and_writes_a_manifest(tmp_path: Path):
    trace = {
        "task_id": "bug-001",
        "attempts": [{
            "attempt_number": 1, "state": "SUCCEEDED", "hypothesis": "h", "patch": "P",
            "model_output": "token ghp_" + "a" * 36 + "\nP",
            "prompt": {"task_id": "bug-001", "issue_description": "i", "failure_trace": "f",
                       "target_file": "a.py", "file_content": "c"},
        }],
    }
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    for n in range(3):  # identical deterministic trials
        (trace_dir / f"run{n}.json").write_text(json.dumps(trace), encoding="utf-8")

    curator = DatasetCurator(InMemoryMemoryStore())
    sft = tmp_path / "out" / "sft_dataset.jsonl"
    assert curator.export_dataset_jsonl(sft, trace_dir=trace_dir) == 1
    assert "ghp_" not in sft.read_text(encoding="utf-8")

    manifest = curator.write_manifest(sft.with_name("manifest.json"), [sft], excluded_tasks={"bug-009"})
    assert manifest["files"]["sft_dataset.jsonl"]["rows"] == 1
    assert manifest["excluded_held_out_tasks"] == 1
    assert manifest_mismatch(sft) is None
    sft.write_text(sft.read_text(encoding="utf-8") + "\n{}", encoding="utf-8")
    assert "changed after export" in manifest_mismatch(sft)


# --- CPU smoke run of the real training loop ----------------------------------------------------

def _tiny_qwen(path: Path) -> None:
    """Save a randomly initialized 2-layer Qwen2 model and a word-level ChatML tokenizer to `path`."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    words = sorted({w for text in ["sys bug fix the guard return x if not items else none ok"] for w in text.split()})
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "[UNK]"]
    vocab = {tok: i for i, tok in enumerate(specials + words + ["system", "user", "assistant", "\n"])}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Split(pattern=r"(<\|im_start\|>|<\|im_end\|>|\n| )", behavior="isolated")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                        unk_token="[UNK]")
    tokenizer.chat_template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    tokenizer.save_pretrained(path)
    config = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                         tie_word_embeddings=True)
    Qwen2ForCausalLM(config).save_pretrained(path)


def test_training_loop_runs_on_cpu_and_saves_an_adapter(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    import torch
    from evoagent.learning.finetune import QLoRAConfig, completion_only_loss, merge_adapter, train_adapter

    model_dir = tmp_path / "tiny"
    _tiny_qwen(model_dir)
    data = tmp_path / "sft.jsonl"
    rows = [row("fix the guard", "train"), row("return x if not items else none", "train"), row("ok", "val")]
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    summary = train_adapter(QLoRAConfig(
        base_model=str(model_dir), dataset_path=data, output_dir=tmp_path / "adapter", epochs=2,
        quantize_4bit=False, gradient_accumulation_steps=1, max_seq_length=64,
        learning_rate=1e-3,
    ), log=lambda _: None)

    assert summary["train_examples"] == 2 and summary["val_examples"] == 1
    assert summary["eval_loss"] is not None and summary["train_loss"] > 0
    assert (tmp_path / "adapter" / "adapter_model.safetensors").exists()
    assert json.loads((tmp_path / "adapter" / "training_summary.json").read_text())["hardware"]["device"] == "cpu"

    # The completion-only loss equals the standard causal-LM loss restricted to the reply
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    feature = tokenize_example(AutoTokenizer.from_pretrained(str(model_dir)), rows[0], 64)
    inputs = {k: torch.tensor([v]) for k, v in feature.items()}
    with torch.no_grad():
        fast = completion_only_loss(model, inputs["input_ids"], inputs["attention_mask"], inputs["labels"])
        reference = model(**inputs).loss
    assert torch.allclose(fast, reference, atol=1e-5)

    merged = merge_adapter(tmp_path / "adapter", tmp_path / "merged", base_model=str(model_dir))
    assert (merged / "config.json").exists()
