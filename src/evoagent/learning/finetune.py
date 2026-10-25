"""
Q-LoRA adapter training (Layer C) sized for a 4 GB GPU such as the GeForce GTX 1650.

torch, transformers and peft are imported inside the functions that need them, so dataset
preparation and the hardware checks are importable and testable without the training extras.
"""

import hashlib
import inspect
import json
from pathlib import Path
import platform
import statistics
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence
from pydantic import BaseModel, Field

IGNORE_INDEX = -100
# Every linear projection of a Llama/Qwen-style decoder. The QLoRA paper found that adapting all of
# them, not only q/v, is needed to match full fine-tuning quality.
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# The repair loop is served the instruct model (Ollama's qwen2.5-coder tags are instruct builds), so
# the adapter must be trained on the same model and chat template
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MIN_VRAM_GB = 3.5


class TrainingDataError(ValueError):
    """The dataset cannot produce a single usable training example."""


class QLoRAConfig(BaseModel):
    """Hyperparameters for one adapter run. Defaults fit Qwen2.5-Coder-1.5B on a 4 GB Turing GPU."""

    base_model: str = DEFAULT_BASE_MODEL
    dataset_path: Path = Path("data/training_splits/sft_dataset.jsonl")
    output_dir: Path = Path("model_weights/evoagent_adapter")
    epochs: float = 3.0
    learning_rate: float = 2e-4
    # Trace prompts include the file under repair and were measured at 562-1060 tokens, so the old
    # 512 limit dropped every example. Longer examples are skipped, never truncated.
    max_seq_length: int = 1536
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = Field(default_factory=lambda: list(DEFAULT_TARGET_MODULES))
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    max_grad_norm: float = 0.3
    weight_decay: float = 0.0
    seed: int = 42
    quantize_4bit: bool = True
    # Turing GPUs (GTX 16xx, RTX 20xx) have no bf16; fp32 is the fallback if fp16 losses turn NaN
    precision: Literal["fp16", "bf16", "fp32"] = "fp16"
    gradient_checkpointing: bool = True
    # Compute the vocabulary projection only where a completion token is predicted. With Qwen's
    # 152k-token vocabulary, full-sequence fp32 logits at 1536 tokens take ~0.9 GB on their own.
    completion_only_logits: bool = True


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def rows_for_split(rows: Sequence[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    """Rows the exporter assigned to `split`; rows without a split are treated as train."""
    return [row for row in rows if (row.get("split") or "train") == split]


def chat_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    """The conversation the model saw. Trace exports carry it; episode-based rows only have a prompt."""
    return list(row.get("messages") or [{"role": "user", "content": row["prompt"]}])


def tokenize_example(tokenizer: Any, row: Dict[str, Any], max_seq_length: int) -> Optional[Dict[str, List[int]]]:
    """
    Tokenize one row with the model's chat template, as it is rendered at inference, and mask the
    prompt so only the assistant reply is learned. Returns None if the example is too long.
    """
    messages = chat_messages(row)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": row["completion"]}], tokenize=False
    )
    if not full_text.startswith(prompt_text):
        raise TrainingDataError(
            "The chat template does not render the prompt as a prefix of the full conversation, so the "
            "reply cannot be separated from the prompt. Use an instruct model with a standard template."
        )
    # Tokenizing the two parts separately keeps the boundary exact; the reply includes the end-of-turn token
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
    input_ids = prompt_ids + completion_ids
    if not completion_ids or len(input_ids) > max_seq_length:
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + completion_ids,
    }


def tokenize_rows(
    tokenizer: Any, rows: Sequence[Dict[str, Any]], max_seq_length: int
) -> tuple[List[Dict[str, List[int]]], List[int]]:
    """Tokenized examples that fit, plus the token length of every row (for the length report)."""
    features, lengths = [], []
    for row in rows:
        feature = tokenize_example(tokenizer, row, max_seq_length=10**9)
        if feature is None:
            continue
        lengths.append(len(feature["input_ids"]))
        if lengths[-1] <= max_seq_length:
            features.append(feature)
    return features, lengths


def pad_batch(features: Sequence[Dict[str, List[int]]], pad_token_id: int) -> Dict[str, List[List[int]]]:
    """Right-pad a batch to its longest example; padded positions are masked and ignored by the loss."""
    width = max(len(f["input_ids"]) for f in features)
    batch: Dict[str, List[List[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        pad = width - len(f["input_ids"])
        batch["input_ids"].append(f["input_ids"] + [pad_token_id] * pad)
        batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
        batch["labels"].append(f["labels"] + [IGNORE_INDEX] * pad)
    return batch


def length_summary(lengths: Sequence[int], max_seq_length: int) -> Dict[str, Any]:
    """Token-length distribution, so --max-seq-length can be chosen from data instead of guessed."""
    if not lengths:
        return {"examples": 0, "fit": 0}
    ordered = sorted(lengths)
    return {
        "examples": len(ordered),
        "fit": sum(1 for n in ordered if n <= max_seq_length),
        "min": ordered[0],
        "median": int(statistics.median(ordered)),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "max": ordered[-1],
    }


def check_gpu(name: str, total_memory_bytes: int, capability: tuple[int, int], precision: str) -> List[str]:
    """Problems and adjustments for the detected GPU; an entry starting with 'ERROR' stops training."""
    notes = []
    vram_gb = total_memory_bytes / 1024**3
    if vram_gb < MIN_VRAM_GB:
        notes.append(f"ERROR: {name} has {vram_gb:.1f} GB VRAM; a 1.5B model in 4-bit needs about {MIN_VRAM_GB} GB.")
    if capability < (8, 0):
        if precision == "bf16":
            notes.append(f"ERROR: {name} (compute capability {capability[0]}.{capability[1]}) has no bf16; use fp16.")
        notes.append("FlashAttention 2 needs compute capability 8.0+; using PyTorch SDPA attention.")
    return notes


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_mismatch(dataset_path: Path) -> Optional[str]:
    """A warning if the dataset no longer matches the manifest written at export time."""
    manifest_file = Path(dataset_path).with_name("manifest.json")
    if not manifest_file.exists():
        return None
    entry = json.loads(manifest_file.read_text(encoding="utf-8")).get("files", {}).get(Path(dataset_path).name)
    if entry and entry.get("sha256") != file_sha256(dataset_path):
        return f"{dataset_path} changed after export (sha256 differs from {manifest_file}); re-run export-dataset."
    return None


def _filter_kwargs(cls: Any, kwargs: Dict[str, Any], log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Keep the arguments this transformers version accepts, mapping renamed ones and reporting the rest."""
    params = inspect.signature(cls.__init__).parameters
    kwargs = dict(kwargs)
    if "eval_strategy" in kwargs and "eval_strategy" not in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    # transformers 5 dropped warmup_ratio; warmup_steps takes a float in [0, 1) as a ratio instead
    if "warmup_ratio" in kwargs and "warmup_ratio" not in params:
        kwargs["warmup_steps"] = kwargs.pop("warmup_ratio")
    dropped = sorted(set(kwargs) - set(params))
    if dropped:
        log(f"[warning] this transformers version ignores: {', '.join(dropped)}")
    return {key: value for key, value in kwargs.items() if key in params}


def _dtype_key() -> str:
    """transformers 4.56 renamed from_pretrained's torch_dtype argument to dtype."""
    from transformers import PreTrainedModel

    return "dtype" if "dtype" in inspect.signature(PreTrainedModel.from_pretrained).parameters else "torch_dtype"


def completion_only_loss(model: Any, input_ids: Any, attention_mask: Any, labels: Any) -> Any:
    """
    Mean cross-entropy over completion tokens, projecting to the vocabulary only at the positions
    that predict one instead of over the whole sequence.
    """
    import torch.nn.functional as F

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = base.get_decoder()(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    # Position t predicts token t+1
    targets = labels[:, 1:]
    mask = targets != IGNORE_INDEX
    logits = base.get_output_embeddings()(hidden[:, :-1][mask]).float()
    return F.cross_entropy(logits, targets[mask])


def train_adapter(config: QLoRAConfig, log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Train a LoRA adapter on the SFT export and save it with a training_summary.json."""
    try:
        import torch
        import transformers
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError(
            f"Training dependencies are missing ({exc}). Install them with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            '  pip install -e ".[training]"'
        ) from exc

    dataset_path = Path(config.dataset_path)
    if not dataset_path.exists():
        raise TrainingDataError(f"{dataset_path} not found. Run `evoagent export-dataset` first.")
    warning = manifest_mismatch(dataset_path)
    if warning:
        log(f"[warning] {warning}")

    rows = load_jsonl(dataset_path)
    # val/test tasks must stay unseen; val only measures loss on tasks the adapter never trained on
    train_rows, val_rows = rows_for_split(rows, "train"), rows_for_split(rows, "val")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_features, train_lengths = tokenize_rows(tokenizer, train_rows, config.max_seq_length)
    val_features, _ = tokenize_rows(tokenizer, val_rows, config.max_seq_length)
    lengths = length_summary(train_lengths, config.max_seq_length)
    log(f"Train examples: {lengths}")
    if not train_features:
        raise TrainingDataError(
            f"None of {len(train_rows)} train rows fits in {config.max_seq_length} tokens "
            f"(lengths: {lengths}). Raise --max-seq-length or collect more traces."
        )

    cuda = torch.cuda.is_available()
    gpu: Dict[str, Any] = {"device": "cpu"}
    precision = config.precision
    if cuda:
        props = torch.cuda.get_device_properties(0)
        capability = torch.cuda.get_device_capability(0)
        gpu = {"device": props.name, "vram_gb": round(props.total_memory / 1024**3, 2), "capability": capability}
        notes = check_gpu(props.name, props.total_memory, capability, precision)
        for note in notes:
            log(note)
        if any(note.startswith("ERROR") for note in notes):
            raise RuntimeError("GPU check failed; see the messages above.")
    elif config.quantize_4bit:
        raise RuntimeError("4-bit training needs a CUDA GPU. Pass --no-4bit for a CPU smoke run.")
    else:
        precision = "fp32"
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]

    model_kwargs: Dict[str, Any] = {"attn_implementation": "sdpa"}
    dtype_key = _dtype_key()
    if config.quantize_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": 0}
        model_kwargs[dtype_key] = dtype
    else:
        # Unquantized LoRA keeps fp32 master weights; AMP handles fp16/bf16 compute on a GPU
        model_kwargs[dtype_key] = torch.float32

    log(f"Loading {config.base_model} ({'4-bit NF4' if config.quantize_4bit else 'unquantized'}, {precision})...")
    model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    model.config.use_cache = False
    if config.quantize_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")

    def collate(features: Sequence[Dict[str, List[int]]]) -> Dict[str, Any]:
        return {k: torch.tensor(v) for k, v in pad_batch(features, tokenizer.pad_token_id).items()}

    class CompletionOnlyTrainer(Trainer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # compute_loss returns a per-batch mean, so let the Trainer divide by the accumulation steps
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
            labels = inputs.pop("labels")
            if config.completion_only_logits:
                loss = completion_only_loss(model, inputs["input_ids"], inputs["attention_mask"], labels)
            else:
                loss = model(**inputs, labels=labels).loss
            return (loss, {"loss": loss}) if return_outputs else loss

    output_dir = Path(config.output_dir)
    args = TrainingArguments(**_filter_kwargs(TrainingArguments, {
        "output_dir": str(output_dir),
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "fp16": cuda and precision == "fp16",
        "bf16": cuda and precision == "bf16",
        "optim": "paged_adamw_8bit" if config.quantize_4bit else "adamw_torch",
        "logging_steps": 5,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "eval_strategy": "epoch" if val_features else "no",
        "prediction_loss_only": True,
        "remove_unused_columns": False,
        "report_to": "none",
        "seed": config.seed,
        "use_cpu": not cuda,
        "dataloader_pin_memory": cuda,
    }, log))
    trainer = CompletionOnlyTrainer(
        model=model,
        args=args,
        train_dataset=train_features,
        eval_dataset=val_features or None,
        data_collator=collate,
    )

    log("Starting fine-tuning...")
    train_output = trainer.train()
    eval_loss = trainer.evaluate()["eval_loss"] if val_features else None

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    summary = {
        "config": json.loads(config.model_dump_json()),
        "dataset_sha256": file_sha256(dataset_path),
        "train_examples": len(train_features),
        "train_rows_skipped_too_long": len(train_rows) - len(train_features),
        "val_examples": len(val_features),
        "token_lengths": lengths,
        "train_loss": train_output.training_loss,
        "eval_loss": eval_loss,
        "trainable_parameters": trainable,
        "hardware": gpu,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log(f"Adapter saved to {output_dir}")
    return summary


def merge_adapter(adapter_dir: Path, output_dir: Path, base_model: Optional[str] = None) -> Path:
    """
    Merge a trained adapter into an fp16 copy of its base model, for conversion to GGUF and serving
    through llama.cpp or Ollama. Runs on the CPU; a 1.5B model needs about 4 GB of RAM.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(adapter_dir)
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    base_model = base_model or adapter_config["base_model_name_or_path"]
    model = AutoModelForCausalLM.from_pretrained(base_model, **{_dtype_key(): torch.float16})
    merged = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    output_dir = Path(output_dir)
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(str(output_dir))
    return output_dir
