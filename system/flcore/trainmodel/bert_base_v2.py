import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertTokenizer, BertModel
import json
from collections import defaultdict

class BERTPromptLearner(nn.Module):
    """
    1) 构造所有 Prompt (pos/neg) => self.all_prompts
    2) tokenize => 得到 self.tokenized_prompts
    3) forward() 时，对正样本前 n_ctx 个 token 用 ctx_global 替换，负样本保留
    4) 最终输出 shape=(batch_total, seq_len, hidden_dim)，给下游 BERT 做 inputs_embeds
    """
    def __init__(
        self,
        n_ctx,
        classnames,
        pretrained_model_name="bert-base-uncased",
        CSC=True,
        random_init=True,
        manual_prompt=True,
        negative_class=False,
        dataset=None,
        LLM_prompt_file='LLM_prompts',
        LLM_prompt_number=-1
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.manual_prompt = manual_prompt
        self.negative_class = negative_class
        self.CSC = CSC
        self.random_init = random_init

        # 初始化 Tokenizer/BertModel 用于后续计算 embeddings
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.bert_model_for_emb = BertModel.from_pretrained(pretrained_model_name)
        self.bert_model_for_emb.eval()  # 用来获取 embeddings, 不训练

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # todo: 支持多GPU
        # self.bert_model_for_emb.to(self.device)

        self.classnames = classnames
        self.n_cls = len(classnames)
        self.pos_prompts_per_class = 3 if not manual_prompt else 1  # 如果 manual_prompt=False，默认为3条描述
        self.neg_prompts_per_class = 1 if negative_class else 0


        print(f'='*50)
        print(f'Prompt File:{LLM_prompt_file}; Prompt number:{LLM_prompt_number}')
        print(f'='*50)

        # --------- 1) 生成所有 Prompt (pos + neg)，并记录到 prompt_index_map ---------
        all_prompts = []
        prompt_index_map = []  # 用于记录每个类别在 all_prompts 中的起始下标、正负样本数量

        if dataset is not None:
            dataset_core = dataset.split('_')[0]
            prompt_dir = f'../dataset/{LLM_prompt_file}/{dataset_core}/text_encoder_prompts.json'
            with open(prompt_dir, 'r') as f:
                text_encoder_prompts = json.load(f)
        else:
            text_encoder_prompts = None

        for i, cname in enumerate(classnames):
            c_proc = cname.replace("_", " ")

            # 正样本
            pos_prompts = []
            if self.manual_prompt:
                # 只有1条 => "A photo of a <classname>."
                pos_prompts = [f"A photo of a {c_proc}."]
                n_pos = 1
            else:
                # 从 text_encoder_prompts 里取 Fine-grained Descriptions
                if LLM_prompt_number > 0:
                    desc_list = text_encoder_prompts[cname]["Fine-grained Descriptions"][0:LLM_prompt_number]
                else:
                    desc_list = text_encoder_prompts[cname]["Fine-grained Descriptions"]
                pos_prompts = [f"A photo of a {c_proc}: {desc}" for desc in desc_list]
                print(f'[{cname}] {pos_prompts}')
                n_pos = len(desc_list)  # 通常=3

            # 负样本
            n_neg = 0
            neg_prompts = []
            if self.negative_class:
                # 每类取1条 Hard Negative
                hard_neg_info = text_encoder_prompts[cname]["Hard Negatives"][0]
                neg_class_name = hard_neg_info["NegativeClassName"]
                neg_text = hard_neg_info["Negative Prompt"]
                neg_prompt = f"A photo of a {neg_class_name}: {neg_text}"
                print(f'[{cname}] {neg_prompt}')
                neg_prompts = [neg_prompt]
                n_neg = 1

            # 整合
            start_idx = len(all_prompts)
            all_prompts.extend(pos_prompts)
            all_prompts.extend(neg_prompts)

            # 记录
            prompt_index_map.append({
                "pos_start": start_idx,
                "pos_count": n_pos,
                "neg_start": (start_idx + n_pos) if n_neg > 0 else None,
                "neg_count": n_neg
            })

        self.prompt_index_map = prompt_index_map
        self.all_prompts = all_prompts
        self.batch_total = len(all_prompts)  # 总的 prompt 数量

        print(f'prompt_index_map: {self.prompt_index_map}')
        print(f'batch_total: {self.batch_total}')

        # --------- 2) tokenize => self.tokenized_prompts (batch_total, seq_len) ---------
        tokenized = self.tokenizer(all_prompts, return_tensors="pt", padding=True, truncation=True)
        self.register_buffer("tokenized_prompts", tokenized["input_ids"])  # shape=(batch_total, seq_len)
        # 注意: 如果后面需要 attention_mask，也可同样存下

        # --------- 3) 可学习上下文 ctx_global 初始化 ---------
        #    不再预先生成 embeddings (因为要在 forward 里按需计算 & 替换)
        hidden_dim = self.bert_model_for_emb.config.hidden_size
        self.ctx_global = None

        if self.n_ctx > 0:
            if self.CSC:
                # class-specific => shape=(n_cls, n_ctx, hidden_dim)
                if random_init:
                    ctx_vectors = torch.empty(self.n_cls, self.n_ctx, hidden_dim)
                    nn.init.normal_(ctx_vectors, std=0.02)
                else:
                    # 简化写法：随机/或别的初始化
                    ctx_vectors = torch.empty(self.n_cls, self.n_ctx, hidden_dim)
                    nn.init.normal_(ctx_vectors, std=0.02)
                self.ctx_global = nn.Parameter(ctx_vectors)
            else:
                # shared => shape=(n_ctx, hidden_dim)
                if random_init:
                    ctx_vectors = torch.empty(self.n_ctx, hidden_dim)
                    nn.init.normal_(ctx_vectors, std=0.02)
                else:
                    ctx_vectors = torch.empty(self.n_ctx, hidden_dim)
                    nn.init.normal_(ctx_vectors, std=0.02)
                self.ctx_global = nn.Parameter(ctx_vectors)

    def forward(self):
        """
        返回一个张量: shape=(batch_total, seq_len, hidden_dim)
          - 对正样本(前 n_ctx)替换为 ctx_global
          - 负样本保持原 embedding
        """

        # 1) 先用 bert_model_for_emb.embeddings() 对所有 tokenized_prompts 做 embedding
        # token_ids = self.tokenized_prompts.to(self.device)  # (batch_total, seq_len)
        token_ids = self.tokenized_prompts
        with torch.no_grad():
            base_emb = self.bert_model_for_emb.embeddings(input_ids=token_ids)  # (batch_total, seq_len, hidden_dim)

        # 2) 克隆一份
        prompts_with_ctx = base_emb.clone()  # (batch_total, seq_len, dim)

        # 3) 替换正样本前 n_ctx token
        if self.ctx_global is not None and self.n_ctx > 0:
            for class_id, info in enumerate(self.prompt_index_map):
                pos_start = info["pos_start"]
                pos_count = info["pos_count"]

                # 负样本数
                n_neg = info["neg_count"]

                # 取出当前类的 ctx_global
                if self.CSC:
                    # shape=(n_ctx, dim)
                    ctx_vec = self.ctx_global[class_id]
                else:
                    # shared => shape=(n_ctx, dim)
                    ctx_vec = self.ctx_global

                # 替换正样本(每条)
                for i in range(pos_count):
                    row_idx = pos_start + i
                    # prompts_with_ctx[row_idx, :n_ctx, :] = ctx_vec
                    # 注：上面一行 = shape mismatch if ctx_vec.dim()==2 => (n_ctx, dim)
                    prompts_with_ctx[row_idx, : self.n_ctx, :] = ctx_vec

                # 负样本 => 不做替换，保持原embedding
                # if n_neg>0, row_idx = pos_start+pos_count ... 不替换

        return prompts_with_ctx


class TextEncoder_server_bert(nn.Module):
    """
    思路：
     - get_text_prototypes() 一次性对 batch_total 条 prompt 前向 => (batch_total, hidden_dim)
     - 根据 prompt_index_map，把正样本(多条)平均，负样本单独取 => 得到 (n_cls)或(2*n_cls, hidden_dim)
    """
    def __init__(
        self,
        classnames,
        n_ctx,
        pretrained_model_name="bert-base-uncased",
        CSC=True,
        random_init=True,
        manual_prompt=True,
        negative_class=False,
        dataset=None,
        LLM_prompt_file='LLM_prompts',
        LLM_prompt_number=-1
    ):
        super().__init__()
        # BERTPromptLearner 负责生成 embedding (含可学习上下文)
        self.prompt_learner = BERTPromptLearner(
            n_ctx,
            classnames,
            pretrained_model_name,
            CSC,
            random_init,
            manual_prompt,
            negative_class,
            dataset,
            LLM_prompt_file=LLM_prompt_file,
            LLM_prompt_number=LLM_prompt_number
        )
        self.bert_model = BertModel.from_pretrained(pretrained_model_name)
        self.bert_model.eval()  # 如果你只在这里做推理，可 eval(); 若要finetune则改成train()

        self.logit_scale = nn.Parameter(torch.tensor(4.6052))  # 你也可初始化为 4.6052 等
        self.num_classes = len(classnames)
        self.negative_class = negative_class

        # 记录 prompt_index_map 用于 get_text_prototypes() 计算
        self.prompt_index_map = self.prompt_learner.prompt_index_map

    def get_text_prototypes(self, training=False):
        """
        1) prompt_learner() => (batch_total, seq_len, hidden_dim)
        2) 过 bert_model => 得到 (batch_total, seq_len, hidden_dim)
        3) 取CLS向量 or pooler_output or 自行池化 => (batch_total, hidden_dim)
        4) 按 prompt_index_map 做正样本平均 & 负样本独立 => (n_cls or 2*n_cls, hidden_dim)
        """

        # 1) 先得到加了可学习上下文的 token-level embedding
        if training:
            # 训练阶段 => 可以产生梯度
            prompts_with_ctx = self.prompt_learner()
        else:
            # 推理 => no_grad
            with torch.no_grad():
                prompts_with_ctx = self.prompt_learner()

        # 2) 用 bert_model(inputs_embeds=...) 得到输出
        if training:
            outputs = self.bert_model(inputs_embeds=prompts_with_ctx)  # (batch_total, seq_len, dim)
            text_features_all = outputs.last_hidden_state  # shape=(batch_total, seq_len, hidden_dim)
        else:
            with torch.no_grad():
                outputs = self.bert_model(inputs_embeds=prompts_with_ctx)
                text_features_all = outputs.last_hidden_state.detach()

        # 这里取 `[CLS]` 位置向量 => text_features_all[:, 0, :]
        # （你也可以改为 average pooling 等）
        text_features_all = text_features_all[:, 0, :]  # (batch_total, hidden_dim)

        # 3) 根据 prompt_index_map，将正样本合并平均 & 负样本单独取
        text_features_pos = []
        text_features_neg = []

        for class_id, info in enumerate(self.prompt_index_map):
            pos_start = info["pos_start"]
            pos_count = info["pos_count"]
            neg_start = info["neg_start"]
            neg_count = info["neg_count"]

            # 取正样本
            pos_slice = text_features_all[pos_start : pos_start + pos_count]  # (pos_count, hidden_dim)
            pos_mean = pos_slice.mean(dim=0, keepdim=True)  # (1, hidden_dim)
            text_features_pos.append(pos_mean)

            # 取负样本
            if neg_count > 0:
                neg_slice = text_features_all[neg_start : neg_start + neg_count]  # (neg_count, hidden_dim)
                # 如果 neg_count=1 就是一条
                neg_mean = neg_slice.mean(dim=0, keepdim=True)
                text_features_neg.append(neg_mean)

        text_features_pos = torch.cat(text_features_pos, dim=0)  # (n_cls, hidden_dim)
        if self.negative_class:
            text_features_neg = torch.cat(text_features_neg, dim=0)  # (n_cls, hidden_dim)
            # 合并 => (2*n_cls, hidden_dim)
            text_features = torch.cat([text_features_pos, text_features_neg], dim=0)
        else:
            text_features = text_features_pos

        return text_features  # (n_cls or 2*n_cls, hidden_dim)

    def forward(self, global_vision_prototype):
        """
        演示与视觉特征做相似度计算的流程
        1) image_features -> L2 normalize
        2) text_features -> get_text_prototypes() & L2 normalize
        3) logit_scale.exp() * (image_features @ text_features.t())
        """
        if isinstance(global_vision_prototype, defaultdict):
            feature_list = [global_vision_prototype[key] for key in range(self.num_classes)]
            image_features = torch.stack(feature_list)
        else:
            image_features = global_vision_prototype

        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 获取文本向量（若要训练 ctx_global 就设置 training=True）
        text_features = self.get_text_prototypes(training=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()  # (n_imgs, n_texts)
        return logits