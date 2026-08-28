"""Config for qwen4_exp (Qwen3.8-Flash-Next).

Every module in this package builds against this dataclass. Derived quantities are
computed once here so no module re-derives them (and gets them subtly wrong).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelArgs:
    # --- identity -----------------------------------------------------------
    model_type: str = "qwen4_exp"

    # --- core transformer ---------------------------------------------------
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False

    # --- attention (QSA layers) --------------------------------------------
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    rope_theta: float = 1e7
    max_position_embeddings: int = 262144
    full_attention_interval: int = 4
    layer_types: Optional[List[str]] = None

    # --- QSA indexer --------------------------------------------------------
    indexer_n_heads: int = 4
    indexer_head_dim: int = 128
    indexer_kv_heads: int = 1
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4

    # --- hyper-connections --------------------------------------------------
    hc_count: int = 4
    hc_lowrank: int = 320

    # --- MoE ----------------------------------------------------------------
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = False

    # --- gated delta net ----------------------------------------------------
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_num_value_heads: int = 48
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # --- PLE n-gram embedding ----------------------------------------------
    ple_layer_ids: List[int] = field(default_factory=lambda: [2])  # 1-BASED in HF config
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ple_embed_dim: int = 2560
    ple_conv_kernel_size: int = 4
    ngram_vocab_size_base: int = 20_000_000
    split_ngram_parts: int = 128

    # --- rope ---------------------------------------------------------------
    rope_parameters: Dict[str, Any] = field(default_factory=dict)

    # --- multimodal / special ids ------------------------------------------
    image_token_id: int = 248056
    video_token_id: int = 248057
    eos_token_id: int = 248044

    # --- MTP speculative decoding ------------------------------------------
    # Set to True to load the mtp.* weights (disabled by default so the non-MTP
    # path stays byte-identical and the checkpoint size is unchanged).
    load_mtp: bool = False

    # ------------------------------------------------------------------ derived
    def __post_init__(self):
        if self.layer_types is None:
            # layer i is full attention iff (i+1) % full_attention_interval == 0
            self.layer_types = [
                "full_attention"
                if (i + 1) % self.full_attention_interval == 0
                else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries, "
                f"expected {self.num_hidden_layers}"
            )

    # -- hyper connections
    @property
    def hc_dim(self) -> int:
        return self.hc_count * self.hidden_size

    # -- gated delta net
    @property
    def key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim  # 2048

    @property
    def value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim  # 6144

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim  # 10240

    # -- PLE
    @property
    def ple_layers(self) -> List[int]:
        """0-based layer indices carrying a PLE block."""
        return [i - 1 for i in self.ple_layer_ids]

    @property
    def ple_n_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram  # 16

    @property
    def ple_head_dim(self) -> int:
        return self.ple_embed_dim // self.ple_n_heads  # 160

    @property
    def ple_conv_history(self) -> int:
        """Timesteps of history the dilated depthwise conv needs."""
        return (self.ple_conv_kernel_size - 1) * self.ngram_size  # 9

    # -- rope
    @property
    def n_rot(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)  # 64

    @property
    def mrope_section(self) -> List[int]:
        return self.rope_parameters.get("mrope_section", [11, 11, 10])

    @property
    def mrope_interleaved(self) -> bool:
        return bool(self.rope_parameters.get("mrope_interleaved", True))

    def is_linear(self, i: int) -> bool:
        return self.layer_types[i] == "linear_attention"

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ModelArgs":
        """Accepts either a full multimodal config.json or a bare text_config."""
        import inspect

        text = dict(params.get("text_config", params))
        # a few ids live at the root of the multimodal config
        for k in ("image_token_id", "video_token_id", "model_type"):
            if k in params and k not in text:
                text[k] = params[k]
        # the root model_type is the multimodal one; the text one is qwen4_exp_text
        text["model_type"] = "qwen4_exp"

        eos = text.get("eos_token_id")
        if isinstance(eos, list):
            text["eos_token_id"] = int(eos[-1])

        allowed = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in text.items() if k in allowed})
