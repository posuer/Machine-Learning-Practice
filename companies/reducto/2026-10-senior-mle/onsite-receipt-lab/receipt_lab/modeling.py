import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from .common import messages


def processor_for(config, source=None):
    processor = AutoProcessor.from_pretrained(
        source or config["model_id"],
        revision=None if source else config.get("model_revision"),
        min_pixels=config["min_pixels"], max_pixels=config["max_pixels"],
    )
    processor.tokenizer.padding_side = "right"
    return processor


def compute_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_model(config):
    kwargs = dict(revision=config.get("model_revision"), torch_dtype=compute_dtype(),
                  attn_implementation="sdpa")
    if config.get("load_in_4bit"):
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit baseline requires CUDA. Use --smoke-model for the CPU wiring test.")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype(),
        )
        kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config["model_id"], **kwargs)
    if not config.get("load_in_4bit"):
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def tiny_model(processor):
    # Random weights: exercises the same multimodal architecture, not OCR quality.
    config = Qwen2_5_VLConfig(
        vocab_size=len(processor.tokenizer), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=4096, tie_word_embeddings=True,
        rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        bos_token_id=151643, eos_token_id=151645, pad_token_id=processor.tokenizer.pad_token_id,
        image_token_id=151655, video_token_id=151656,
        vision_start_token_id=151652, vision_end_token_id=151653,
        vision_config={"depth": 2, "hidden_size": 64, "intermediate_size": 128,
                       "num_heads": 4, "in_chans": 3, "out_hidden_size": 64,
                       "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2,
                       "window_size": 112, "fullatt_block_indexes": [1]},
    )
    return Qwen2_5_VLForConditionalGeneration(config)


class ReceiptCollator:
    """TRL VLM collator adapted to supervise only the JSON answer (and its EOS).

    Keep original-position labels: Qwen's forward computes the causal shift internally.
    Never truncate multimodal token sequences independently of image_grid_thw.
    """
    def __init__(self, processor, max_sequence_tokens=4096):
        self.processor = processor
        self.max_sequence_tokens = max_sequence_tokens

    def __call__(self, examples):
        images = []
        for row in examples:
            with Image.open(row["image_path"]) as im:
                images.append(im.convert("RGB"))
        prompts = [self.processor.apply_chat_template(messages(), tokenize=False, add_generation_prompt=True)
                   for _ in examples]
        texts = [self.processor.apply_chat_template(messages(row["target"]), tokenize=False,
                                                    add_generation_prompt=False) for row in examples]
        batch = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        prefixes = self.processor(text=prompts, images=images, padding=True, return_tensors="pt")
        if batch["input_ids"].shape[1] > self.max_sequence_tokens:
            raise ValueError("Sequence exceeds token budget; inspect the sample or revise image/token budget explicitly.")
        labels = batch["input_ids"].clone()
        for index in range(len(examples)):
            length = int(prefixes["attention_mask"][index].sum())
            if not torch.equal(batch["input_ids"][index, :length], prefixes["input_ids"][index, :length]):
                raise ValueError("Chat-template prefix mismatch; cannot safely mask the prompt")
            labels[index, :length] = -100
        labels[batch["attention_mask"] == 0] = -100
        if not (labels[:, 1:] != -100).any(dim=1).all():
            raise ValueError("Sample has no supervised next-token targets")
        batch["labels"] = labels
        return batch
