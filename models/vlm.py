import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from .encoders import build_vision_encoder, VisionEncoderConfig, VisionEncoder
from .decoders import build_text_decoder, TextDecoderConfig, TextDecoder
from .decoders.text.base import DEFAULT_IMAGE_TOKEN
from .projector import PerceiverResampler


class VlmConfig(BaseModel):
    # VLM配置必要字段
    vision_encoder: VisionEncoderConfig  # 视觉编码器
    text_decoder: TextDecoderConfig      # 文本解码器

    # 连接器(视觉→文本)配置:均有默认值,configs 中可省略
    num_image_tokens: int = 64   # 送入文本解码器的视觉token数(重采样查询数)
    projector_layers: int = 2    # 重采样层数
    projector_heads: int = 8     # 注意力头数(需整除文本解码器 hidden_dim)


class VLM(nn.Module):
    def __init__(self, cfg: VlmConfig):
        super().__init__()
        self.cfg = cfg
        self.vision_encoder: VisionEncoder = build_vision_encoder(cfg.vision_encoder)
        self.text_decoder: TextDecoder = build_text_decoder(cfg.text_decoder)

        # 图像占位符token:与文本解码器 setup_image_token 的选择保持一致
        self.image_token: str = cfg.text_decoder.args.get("image_token") or DEFAULT_IMAGE_TOKEN
        self.num_image_tokens: int = cfg.num_image_tokens

        # 连接器:变长视觉序列 (B, N_v, vision_dim) → (B, num_image_tokens, llm_dim)
        vision_dim = getattr(self.vision_encoder, "out_dim", None)
        if vision_dim is None:
            raise AttributeError(
                f"视觉编码器 {type(self.vision_encoder).__name__} 未暴露 out_dim 属性,无法构建连接器")
        llm_dim = self.text_decoder.model.config.hidden_size
        if llm_dim % cfg.projector_heads != 0:
            raise ValueError(f"projector_heads={cfg.projector_heads} 不整除文本解码器 hidden_dim={llm_dim}")
        self.projector = PerceiverResampler(
            vision_dim, llm_dim,
            num_queries=cfg.num_image_tokens,
            num_layers=cfg.projector_layers,
            num_heads=cfg.projector_heads,
        )

    # ================================== 文本处理 ==================================
    def text_tokenlize(self, prompt):
        """字符串转token(batch内不等长时右padding)"""
        if isinstance(prompt, str):
            prompt = [prompt]
        text_feature = self.text_decoder.tokenizer(
            prompt, return_tensors="pt", padding=True, add_special_tokens=False)
        ids, att_mask = text_feature.input_ids, text_feature.attention_mask
        return ids, att_mask

    def ids_embedding(self, ids):
        """文本embedding
            ids: (B, L)
        """
        return self.text_decoder.get_text_embed(ids)

    @staticmethod
    def _normalize_texts(texts, batch_size):
        """str | list[str] | None → 长度 batch_size 的 list[str];None/空串 = 无文本提示"""
        if texts is None:
            return [""] * batch_size
        if isinstance(texts, str):
            return [texts] * batch_size
        texts = ["" if t is None else t for t in texts]
        if len(texts) != batch_size:
            raise ValueError(f"文本数量 {len(texts)} 与 batch 大小 {batch_size} 不一致")
        return texts

    # def _prefix_strings(self, prompts):
    #     """输入前缀 = num_image_tokens 个图像占位符 + 可选文本提示"""
    #     return [self.image_token * self.num_image_tokens + p for p in prompts]

    def _pad_batch(self, prefix_ids, target_ids):
        """逐样本拼接 前缀+目标 后右padding成batch张量。

        返回 (input_ids, attention_mask, labels),形状 (B, L);
        labels 仅目标段有效,前缀(含图像占位符)与padding置 -100。
        """
        tok = self.text_decoder.tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        B = len(prefix_ids)
        L = max(len(p) + len(t) for p, t in zip(prefix_ids, target_ids))
        input_ids = torch.full((B, L), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, L), dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        for i, (p, t) in enumerate(zip(prefix_ids, target_ids)):
            n = len(p)
            input_ids[i, :n] = torch.as_tensor(p, dtype=torch.long)
            input_ids[i, n:n + len(t)] = torch.as_tensor(t, dtype=torch.long)
            attention_mask[i, :n + len(t)] = 1
            labels[i, n:n + len(t)] = torch.as_tensor(t, dtype=torch.long)
        return input_ids, attention_mask, labels

    # ================================== 图像处理 ==================================
    def img_embedding(self, img):
        """ 图像embedding
            img: (B, C, D, H, W)
            返回: (B, num_image_tokens, llm_dim),可直接替换文本占位符
        """
        feats = self.vision_encoder(img)
        if isinstance(feats, tuple):  # 兼容 (tokens, attn_mask) 约定的编码器
            feats, v_mask = feats[0], feats[1]
        else:
            v_mask = None
        return self.projector(feats, v_mask)

    # ================================== 冻结/解冻参数 ==================================
    def freeze(self, module:str = None):
        """冻结参数"""
        if module == "vision_encoder":
            # 冻结视觉编码器
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
        elif module == "projector":
            # 冻结连接器
            for p in self.projector.parameters():
                p.requires_grad = False
        elif module == "text_decoder":
            # 冻结文本解码器
            for p in self.text_decoder.parameters():
                p.requires_grad = False
        elif module is None:
            # 全部冻结
            for p in self.parameters():
                p.requires_grad = False

    def unfreeze(self, module:str = None):
        """解冻参数"""
        if module == "vision_encoder":
            # 解冻视觉编码器
            for p in self.vision_encoder.parameters():
                p.requires_grad = True
        elif module == "projector":
            # 解冻连接器
            for p in self.projector.parameters():
                p.requires_grad = True
        elif module == "text_decoder":
            # 解冻文本解码器
            for p in self.text_decoder.parameters():
                p.requires_grad = True
        elif module is None:
            # 全部解冻
            for p in self.parameters():
                p.requires_grad = True
    # ================================== 前向计算逻辑 ==================================
    def forward(self, img, prompt=None, target=None):
        """
            img: (B, C, D, H, W),已归一化到 [0,1]
            prompt: list(str) | str | None,可选文本提示(问答场景);无提示传 None
            target: list(str) | None,训练目标文本(报告/答案);给出时计算loss
            返回: (logits, loss);logits (B, L, vocab);loss 为标量或 None(未给target)
        """
        B = img.shape[0]
        device = next(self.text_decoder.parameters()).device
        tok = self.text_decoder.tokenizer

        # 视觉embedding (B, num_image_tokens, llm_dim)
        vis_embeds = self.img_embedding(img)

        # 前缀 = 图像占位符 + 可选提示;目标 = target + eos
        prompts = self._normalize_texts(prompt, B)
        prefix_ids = self.text_decoder.build_prefix_ids(
            prompts, self.num_image_tokens, self.image_token)

        if target is not None:
            targets = self._normalize_texts(target, B)
            target_ids = self.text_decoder.build_target_ids(targets)
        else:
            target_ids = [[] for _ in range(B)]

        input_ids, att_mask, labels = self._pad_batch(prefix_ids, target_ids)
        input_ids, att_mask, labels = (input_ids.to(device),
                                       att_mask.to(device), labels.to(device))

        # 文本中图像占位符原位替换为视觉embedding,过LLM
        logits = self.text_decoder(input_ids, vis_embeds, att_mask)

        loss = None
        if target is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    # ================================== 推理生成 ==================================
    @torch.no_grad()
    def generate(self, img, prompt=None, max_new_tokens=128, do_sample=False,
                 temperature=1.0, top_k=50, top_p=1.0, **gen_kwargs):
        """
            img: (B, C, D, H, W)
            prompt: list(str) | str | None,可选文本提示(原始文本,chat格式由调用方拼好)
            自回归生成文本,返回 list[str](长度B)
        """
        was_training = self.training
        self.eval()
        try:
            B = img.shape[0]
            device = next(self.text_decoder.parameters()).device
            tok = self.text_decoder.tokenizer

            vis_embeds = self.img_embedding(img)  # (B, num_image_tokens, llm_dim)

            prompts = self._normalize_texts(prompt, B)
            prefix_ids = self.text_decoder.build_prefix_ids(
                prompts, self.num_image_tokens, self.image_token)

            # 变长 list → 右 padding 成 batch
            pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            L = max(len(p) for p in prefix_ids)
            input_ids = torch.full((B, L), pad_id, dtype=torch.long)
            att_mask = torch.zeros((B, L), dtype=torch.long)
            for i, p in enumerate(prefix_ids):
                input_ids[i, :len(p)] = torch.as_tensor(p, dtype=torch.long)
                att_mask[i, :len(p)] = 1
            input_ids, att_mask = input_ids.to(device), att_mask.to(device)

            # 文本embedding + 视觉embedding原位替换占位符 → inputs_embeds
            text_embeds = self.ids_embedding(input_ids)
            image_mask = input_ids == self.text_decoder.image_pad_id
            n_img = int(image_mask.sum().item())
            assert n_img == vis_embeds.shape[0] * vis_embeds.shape[1], \
                f"图像占位符数量 {n_img} != 视觉token数 {vis_embeds.shape[0] * vis_embeds.shape[1]}"
            vis = vis_embeds.reshape(-1, vis_embeds.shape[-1]).to(
                device=text_embeds.device, dtype=text_embeds.dtype)
            inputs_embeds = text_embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(text_embeds), vis)

            gen = dict(inputs_embeds=inputs_embeds, attention_mask=att_mask,
                       max_new_tokens=max_new_tokens, do_sample=do_sample)
            if do_sample:
                gen.update(temperature=temperature, top_k=top_k, top_p=top_p)
            else:  # 覆盖权重自带generation_config的采样参数,保证贪心解码
                gen.update(temperature=None, top_k=None, top_p=None)

            # 终止符:eos + 模型对话结束符(如 <|im_end|>)
            stop_ids = [tok.eos_token_id]
            for t in ("<|im_end|>", "<|endoftext|>"):
                tid = tok.convert_tokens_to_ids(t)
                if tid is not None and tid != tok.unk_token_id and tid not in stop_ids:
                    stop_ids.append(tid)
            gen["eos_token_id"] = stop_ids

            gen.update(gen_kwargs)
            out_ids = self.text_decoder.model.generate(**gen)
            return self.text_decoder.decode(out_ids)
        finally:
            if was_training:
                self.train()


# if __name__ == "__main__":
#     import json

#     torch.manual_seed(0)

#     with open("/hpc/jtia475/vlm_train/configs/test.json") as fp:
#         cfg_dict = json.load(fp)

#     cfg = VlmConfig(**cfg_dict["model"])
#     print(cfg)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"device: {device}")
#     model = VLM(cfg).to(device).eval()

#     n_proj = sum(p.numel() for p in model.projector.parameters())
#     print(f"projector: {n_proj / 1e6:.2f}M params, {model.num_image_tokens} image tokens/sample")

#     # 伪体数据:[0,1]归一化,空间尺寸需被 patch (4,16,16) 整除
#     B = 2
#     img = torch.rand(B, 1, 32, 128, 128, device=device)

#     # ---------- 1. 训练前向:prompt + target → loss,并验证反传 ----------
#     prompts = ["请描述这份脑部影像。", "这份影像有什么异常?"]
#     targets = ["脑部MRI平扫未见明显异常。", "右侧基底节区可见斑片状异常信号影。"]
#     logits, loss = model(img, prompt=prompts, target=targets)
#     print(f"[train]          logits: {tuple(logits.shape)}, loss: {loss.item():.4f}")
#     model.zero_grad()
#     loss.backward()
#     gnorm = sum(p.grad.norm().item() for p in model.projector.parameters()
#                 if p.grad is not None)
#     print(f"[train]          projector grad norm: {gnorm:.4f}")

#     # ---------- 2. 训练前向:无文本prompt(纯图像→报告) ----------
#     with torch.no_grad():
#         logits, loss = model(img, target=targets)
#     print(f"[train no-prompt] logits: {tuple(logits.shape)}, loss: {loss.item():.4f}")

#     # ---------- 3. 生成:问答(带prompt) / 报告生成(无prompt) ----------
#     answers = model.generate(img, prompt=prompts, max_new_tokens=48)
#     for q, a in zip(prompts, answers):
#         print(f"[QA]     {q} -> {a}")
#     reports = model.generate(img, max_new_tokens=48)
#     for r in reports:
#         print(f"[report] {r}")
