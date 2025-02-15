import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from collections import defaultdict
import json

_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    """
    与原先相同：接收 token-level 的张量 prompts，再加 positional_embedding，过 transformer，
    最后取 EOT 位置做投影。
    """
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        """
        Args:
            prompts: shape = (batch_size, n_ctx, transformer.width)
            tokenized_prompts: shape = (batch_size, seq_len) for argmax to find EOT

        Returns:
            text_features: (batch_size, transformer.width)
        """
        # prompts + positional_embedding
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # (N, L, D) -> (L, N, D) for transformer
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (L, N, D) -> (N, L, D)
        x = self.ln_final(x).type(self.dtype)

        # EOT 索引
        eot_indices = tokenized_prompts.argmax(dim=-1)  # (batch_size,)
        text_features = x[torch.arange(x.shape[0]), eot_indices] @ self.text_projection
        return text_features


class PromptLearner_client(nn.Module):
    """
    修正后的 PromptLearner_client:
    - 对正样本的前 n_ctx 个 token 用 ctx_global 替换
    - 负样本保持原样
    - 保持所有 prompt 的序列长度一致
    """
    def __init__(
        self,
        n_ctx_num,
        classnames,
        clip_model,
        CSC=True,
        random_init=True,
        manual_prompt=True,
        negative_class=False,
        dataset=None,
        LLM_prompt_file='LLM_prompts',
        LLM_prompt_number=-1
    ):
        super().__init__()
        self.n_ctx = n_ctx_num
        self.CSC = CSC
        self.random_init = random_init
        self.manual_prompt = manual_prompt
        self.negative_class = negative_class
        self.clip_model = clip_model

        self.device = next(clip_model.parameters()).device
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        print(f'='*50)
        print(f'Prompt File:{LLM_prompt_file}; Prompt number:{LLM_prompt_number}')
        print(f'='*50)

        # 1) 构造所有文本 Prompt (正样本 + 负样本)，并分别 tokenize
        all_prompts = []
        prompt_index_map = []  # 记录每个类别的正负样本索引

        # 读取 JSON 中的 prompt 数据
        if dataset is not None:
            dataset_core = dataset.split('_')[0]
            prompt_dir = f'../dataset/{LLM_prompt_file}/{dataset_core}/text_encoder_prompts.json'
            with open(prompt_dir, 'r') as f:
                text_encoder_prompts = json.load(f)
        else:
            text_encoder_prompts = None

        for i, cname in enumerate(classnames):
            class_prompts = []
            c_proc = cname.replace("_", " ")

            if self.manual_prompt:
                # 正样本 Prompt: "A photo of a <classname>."
                prompt_str = f"A photo of a {c_proc}."
                class_prompts.append(prompt_str)
                pos_count = 1
            else:
                # 正样本 Prompt: 从 Fine-grained Descriptions 中取出多个描述
                if LLM_prompt_number > 0:
                    desc_list = text_encoder_prompts[cname]["Fine-grained Descriptions"][0:LLM_prompt_number]
                else:
                    desc_list = text_encoder_prompts[cname]["Fine-grained Descriptions"]
                class_prompts = [f"A photo of a {c_proc}: {desc}" for desc in desc_list]
                print(f'[{cname}] {class_prompts}')
                pos_count = len(desc_list)  # 一般为3

            # 负样本 Prompt
            neg_prompts = []
            neg_count = 0
            if self.negative_class:
                hard_neg_info = text_encoder_prompts[cname]["Hard Negatives"][0]  # 取第一条 Hard Negative
                neg_class = hard_neg_info["NegativeClassName"]
                neg_prompt = f"A photo of a {neg_class}: {hard_neg_info['Negative Prompt']}"
                print(f'[{cname}] {neg_prompt}')
                neg_prompts.append(neg_prompt)
                neg_count = 1

            # 汇总正负样本 Prompts
            all_prompts.extend(class_prompts)  # 正样本
            all_prompts.extend(neg_prompts)    # 负样本

            # 记录每个类别的 Prompt 索引
            prompt_index_map.append({
                "pos_start": len(all_prompts) - (pos_count + neg_count),
                "pos_count": pos_count,
                "neg_start": len(all_prompts) - neg_count if neg_count > 0 else None,
                "neg_count": neg_count
            })

        # Tokenize 所有 Prompts
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in all_prompts])  # (batch_total, seq_len)
        self.register_buffer("tokenized_prompts", tokenized_prompts)  # (batch_total, seq_len)
        self.prompt_index_map = prompt_index_map
        self.batch_total = len(all_prompts)
        self.n_cls = len(classnames)

        # 2) 初始化可学习上下文 (ctx_global)
        if self.n_ctx > 0:
            if self.CSC:
                print("[PromptLearner_client] Using class-specific contexts (CSC).")
                if random_init:
                    ctx_vectors = torch.empty(self.n_cls, self.n_ctx, ctx_dim, dtype=self.dtype)
                    nn.init.normal_(ctx_vectors, std=0.02)
                    self.ctx_global = nn.Parameter(ctx_vectors)  # (n_cls, n_ctx, dim)
                else:
                    # 这里可以根据需要初始化 ctx_global，比如从某些固定的 embedding 中提取
                    print("Initializing class-specific with random due to no base embedding alignment.")
                    ctx_vectors = torch.empty(self.n_cls, self.n_ctx, ctx_dim, dtype=self.dtype)
                    nn.init.normal_(ctx_vectors, std=0.02)
                    self.ctx_global = nn.Parameter(ctx_vectors)
            else:
                print("[PromptLearner_client] Using a generic (shared) context.")
                if random_init:
                    ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=self.dtype)
                    nn.init.normal_(ctx_vectors, std=0.02)
                    self.ctx_global = nn.Parameter(ctx_vectors)  # (n_ctx, dim)
                else:
                    # 同上，根据需要初始化
                    print("Initializing shared with random due to no base embedding alignment.")
                    ctx_vectors = torch.empty(self.n_ctx, ctx_dim, dtype=self.dtype)
                    nn.init.normal_(ctx_vectors, std=0.02)
                    self.ctx_global = nn.Parameter(ctx_vectors)
        else:
            self.ctx_global = None

        print(f'prompt_index_map: {self.prompt_index_map}')
        print(f'batch_total: {self.batch_total}')

    def forward(self):
        """
        返回 shape = (batch_total, seq_len, dim)，其中：
        - 对正样本：用 self.ctx_global 替换掉 base_embedding 的前 n_ctx tokens
        - 对负样本：保持原始 embedding
        """
        # 1) 获取 base_embedding
        with torch.no_grad():
            base_emb = self.clip_model.token_embedding(self.tokenized_prompts.to(self.device)).type(self.dtype)  # (batch_total, seq_len, dim)

        # 2) 克隆 base_emb 以避免 in-place 修改
        prompts_with_ctx = base_emb.clone()

        # 3) 替换正样本的前 n_ctx tokens
        if self.ctx_global is not None and self.n_ctx > 0:
            if self.CSC:
                # class-specific context
                for class_id, info in enumerate(self.prompt_index_map):
                    pos_start = info["pos_start"]
                    pos_count = info["pos_count"]
                    ctx = self.ctx_global[class_id]  # (n_ctx, dim)
                    for i in range(pos_count):
                        prompt_idx = pos_start + i
                        prompts_with_ctx[prompt_idx, :self.n_ctx, :] = ctx
            else:
                # shared context
                prompts_with_ctx[:, :self.n_ctx, :] = self.ctx_global.unsqueeze(0).expand(prompts_with_ctx.shape[0], -1, -1)

        return prompts_with_ctx  # (batch_total, seq_len, dim)


class TextEncoder_server(nn.Module):
    """
    修正后的 TextEncoder_server:
    - 使用 PromptLearner_client.forward() 获取 prompts_with_ctx
    - 调用 TextEncoder 进行前向计算
    - 根据 prompt_index_map 计算每个类别的 text_features（正样本平均 + 负样本独立）
    """
    def __init__(
        self,
        classnames,
        clip_model,
        n_ctx_num=16,
        CSC=True,
        random_init=True,
        manual_prompt=True,
        negative_class=False,
        dataset=None,
        LLM_prompt_file='LLM_prompts',
        LLM_prompt_number=-1
    ):
        super().__init__()
        self.prompt_learner = PromptLearner_client(
            n_ctx_num,
            classnames,
            clip_model,
            CSC,
            random_init,
            manual_prompt=manual_prompt,
            negative_class=negative_class,
            dataset=dataset,
            LLM_prompt_file=LLM_prompt_file,
            LLM_prompt_number=LLM_prompt_number
        )
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.num_classes = len(classnames)

        # 记录 prompt_index_map, 便于 get_text_prototypes() 计算
        self.prompt_index_map = self.prompt_learner.prompt_index_map
        self.negative_class = negative_class
        self.pos_prompts_per_class = 1 if manual_prompt else 3

    def get_text_prototypes(self, training=False):
        """
        1) PromptLearner_client.forward() => (batch_total, seq_len, dim)
        2) self.text_encoder => (batch_total, text_emb_dim)
        3) 按照 prompt_index_map，把同一类别的多个正样本特征平均；若有负样本，把它们单独存起来
        4) 最终输出 shape = (2*n_cls, text_emb_dim) if negative_class=True, else (n_cls, text_emb_dim)
        """
        if training:
            prompts_with_ctx = self.prompt_learner()  # (batch_total, seq_len, dim)
            # 2) 用 text_encoder + tokenized_prompts => (batch_total, text_emb_dim)
            text_features_all = self.text_encoder(prompts_with_ctx, self.tokenized_prompts.to(prompts_with_ctx.device))
        else:
            with torch.no_grad():
                # 1) 得到插入上下文后的 token-level embedding
                prompts_with_ctx = self.prompt_learner()  # (batch_total, seq_len, dim)
                # 2) 用 text_encoder + tokenized_prompts => (batch_total, text_emb_dim)
                text_features_all = self.text_encoder(prompts_with_ctx, self.tokenized_prompts.to(prompts_with_ctx.device))
                text_features_all = text_features_all.detach()
        # 3) 根据 prompt_index_map，将正样本合并平均 & 负样本单独取
        text_features_pos = []
        text_features_neg = []

        for class_id, info in enumerate(self.prompt_index_map):
            pos_start = info["pos_start"]
            pos_count = info["pos_count"]
            neg_start = info["neg_start"]
            neg_count = info["neg_count"]

            # 取正样本
            pos_slice = text_features_all[pos_start : pos_start + pos_count]  # (pos_count, text_emb_dim)
            pos_mean = pos_slice.mean(dim=0, keepdim=True)  # (1, text_emb_dim)
            text_features_pos.append(pos_mean)

            # 取负样本
            if neg_count > 0:
                neg_slice = text_features_all[neg_start : neg_start + neg_count]  # (neg_count, text_emb_dim)
                neg_mean = neg_slice.mean(dim=0, keepdim=True)
                text_features_neg.append(neg_mean)

        text_features_pos = torch.cat(text_features_pos, dim=0)  # (n_cls, text_emb_dim)
        if self.negative_class:
            text_features_neg = torch.cat(text_features_neg, dim=0)  # (n_cls, text_emb_dim)
            # 拼成 (2*n_cls, text_emb_dim)
            text_features = torch.cat([text_features_pos, text_features_neg], dim=0)
        else:
            text_features = text_features_pos

        return text_features  # shape=(n_cls或2*n_cls, text_emb_dim)

    def forward(self, global_vision_prototype):
        """
        1) image_features归一化
        2) get_text_prototypes => text_features归一化
        3) logits = logit_scale.exp() * image_features @ text_features.t()
        """
        # 处理图像特征
        if isinstance(global_vision_prototype, defaultdict):
            # 按类别顺序堆叠
            feature_list = [global_vision_prototype[key] for key in range(self.num_classes)]
            image_features = torch.stack(feature_list)
        else:
            image_features = global_vision_prototype

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 得到 text_features
        text_features = self.get_text_prototypes(training=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 相似度
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits