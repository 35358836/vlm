import torch
import torch.nn as nn

class PerceiverResampler(nn.Module):
    """视觉→文本连接器。

    3D编码器输出的token数随体数据尺寸变化且动辄上千,直接拼接会撑爆LLM上下文;
    用固定数量查询token做交叉注意力重采样,得到 (B, num_queries, llm_dim) 的
    定长视觉序列,同时完成 vision_dim → llm_dim 的维度对齐。
    """

    def __init__(self, vision_dim: int, llm_dim: int, num_queries: int = 64,
                 num_layers: int = 2, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, llm_dim) * 0.02)
        self.in_proj = nn.Linear(vision_dim, llm_dim)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ln_self": nn.LayerNorm(llm_dim),
                "self_attn": nn.MultiheadAttention(llm_dim, num_heads, batch_first=True),
                "ln_cross": nn.LayerNorm(llm_dim),
                "cross_attn": nn.MultiheadAttention(llm_dim, num_heads, batch_first=True),
                "ln_mlp": nn.LayerNorm(llm_dim),
                "mlp": nn.Sequential(
                    nn.Linear(llm_dim, int(llm_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Linear(int(llm_dim * mlp_ratio), llm_dim),
                ),
            })
            for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(llm_dim)

    def forward(self, vision_tokens: torch.Tensor, vision_mask=None) -> torch.Tensor:
        """
            vision_tokens: (B, N_v, vision_dim)
            vision_mask: (B, N_v),1=有效;None=全部有效
            返回: (B, num_queries, llm_dim)
        """
        memory = self.in_proj(vision_tokens.to(self.in_proj.weight.dtype))
        key_padding_mask = None
        if vision_mask is not None:  # MHA约定:True=忽略
            key_padding_mask = ~vision_mask.bool().to(memory.device)

        B = memory.shape[0]
        q = self.queries.to(memory.dtype).expand(B, -1, -1)
        for layer in self.layers:
            h = layer["ln_self"](q)
            q = q + layer["self_attn"](h, h, h, need_weights=False)[0]
            h = layer["ln_cross"](q)
            q = q + layer["cross_attn"](h, memory, memory,
                                        key_padding_mask=key_padding_mask,
                                        need_weights=False)[0]
            q = q + layer["mlp"](layer["ln_mlp"](q))
        return self.ln_out(q)
