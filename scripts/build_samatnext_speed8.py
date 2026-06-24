#!/usr/bin/env python3
"""Build and inspect the scratch SamatNext-Speed-8L-56M causal LM."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file


DEFAULT_CONFIG = Path("configs/samatnext_speed8_640.json")


@dataclass
class Speed8Config:
    model_name: str = "SamatNext-Speed-8L-56M"
    num_layers: int = 8
    hidden_size: int = 640
    intermediate_size: int = 1792
    num_attention_heads: int = 10
    num_key_value_heads: int = 2
    head_dim: int = 64
    vocab_size: int = 32768
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0

    @classmethod
    def from_json(cls, path: Path) -> "Speed8Config":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)

    def validate(self) -> None:
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return x * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position_embeddings: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_position_embeddings).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
        return cos[None, None, :, :], sin[None, None, :, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Speed8Attention(nn.Module):
    def __init__(self, config: Speed8Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.dropout_p = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        if self.num_key_value_groups != 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, config: Speed8Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Speed8Block(nn.Module):
    def __init__(self, config: Speed8Config):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = Speed8Attention(config)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x), cos, sin)
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class SamatNextSpeed8LM(nn.Module):
    def __init__(self, config: Speed8Config):
        super().__init__()
        config.validate()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)
        self.layers = nn.ModuleList([Speed8Block(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.tie_weights()

    def tie_weights(self) -> None:
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        return_logits: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        seq_len = input_ids.shape[1]
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(f"seq_len {seq_len} exceeds max_position_embeddings {self.config.max_position_embeddings}")
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(seq_len, x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x) if return_logits else None
        loss = None
        if labels is not None:
            if logits is None:
                raise ValueError("labels require return_logits=True")
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size).float(), labels.reshape(-1))
        return {"loss": loss, "logits": logits, "hidden_states": x if return_hidden_states else None}

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(f"seq_len {seq_len} exceeds max_position_embeddings {self.config.max_position_embeddings}")
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(seq_len, x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


def parameter_count(model: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        data_ptr = parameter.data_ptr()
        if data_ptr in seen:
            continue
        seen.add(data_ptr)
        total += parameter.numel()
    return total


def model_state_for_save(model: SamatNextSpeed8LM) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    if model.config.tie_word_embeddings:
        state = {key: value for key, value in state.items() if key != "lm_head.weight"}
    return {key: value.detach().cpu() for key, value in state.items()}


def save_checkpoint(model: SamatNextSpeed8LM, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(model_state_for_save(model), out_dir / "model.safetensors")
    (out_dir / "config.json").write_text(json.dumps(asdict(model.config), indent=2) + "\n", encoding="utf-8")


def build_model_from_config(path: Path, dtype: torch.dtype | None = None, device: str | torch.device = "cpu") -> SamatNextSpeed8LM:
    config = Speed8Config.from_json(path)
    model = SamatNextSpeed8LM(config)
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--save-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = Speed8Config.from_json(config_path)
    model = SamatNextSpeed8LM(config)
    metadata: dict[str, Any] = {
        "config": asdict(config),
        "parameter_count": parameter_count(model),
        "dtype": str(next(model.parameters()).dtype),
    }
    if args.save_dir:
        save_checkpoint(model, Path(args.save_dir))
        metadata["save_dir"] = str(Path(args.save_dir).resolve())
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
