from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch

from collections import defaultdict
import json


class BERTPromptLearner(nn.Module):
    def __init__(self, n_ctx, classnames, pretrained_model_name="bert-base-uncased", CSC=True, random_init=True, manual_prompt=True, negative_class=False, dataset=None):
        super().__init__()

        # Initialize tokenizer and model temporarily to obtain fixed suffix embeddings
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        bert_model = BertModel.from_pretrained(pretrained_model_name)

        # Define variables based on input arguments
        n_cls = len(classnames)
        ctx_dim = bert_model.config.hidden_size
        self.n_ctx = n_ctx
        self.manual_prompt = manual_prompt
        self.negative_class = negative_class
        self.CSC = CSC  # Class-Specific Contexts

        self.pos_cls_count = n_cls
        self.neg_cls_count = 0

        if self.manual_prompt:
            # Prompt template setup
            prompt_prefix = "A photo of a"
            prompts = [f"{prompt_prefix} {name}." for name in classnames]
            print('[BERTPromptLearner] manual_prompt=True: using prompts like "A photo of a <classname>."')
            # Tokenize prompts to get non-trainable suffix embeddings
            tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                fixed_embeddings = bert_model.embeddings(input_ids=tokenized_prompts["input_ids"]).detach()

        else:
            # obtain LLM generated prompt
            assert dataset is not None
            dataset = dataset.split('_')[0]
            prompt_dir = f'../dataset/rawdata/{dataset}/text_encoder_prompts.json'
            with open(prompt_dir, 'r') as f:
                text_encoder_prompts = json.load(f)
            fine_grained_dict = {}
            for classname in classnames:
                fine_grained_description = text_encoder_prompts[classname]["Fine-grained Descriptions"]
                fine_grained_dict[classname] = fine_grained_description

            print("[BERTPromptLearner] manual_prompt=False: using fine-grained descriptions and averaging embeddings.")
            
            base_embeddings = []
            for cname in classnames:
                desc_list = fine_grained_dict[cname]
                prompts_3 = [f"{cname}: {desc}" for desc in desc_list]

                tokenized_3 = tokenizer(prompts_3, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    emb_3 = bert_model.embeddings(
                        input_ids=tokenized_3["input_ids"].to(self.device)
                    ).detach()  # (3, seq_len, hidden_dim)

                emb_mean = emb_3.mean(dim=0, keepdim=True)
                base_embeddings.append(emb_mean)

            # (n_cls, seq_len, hidden_dim)
            base_embedding = torch.cat(base_embeddings, dim=0)

        # ------------ 2) 如果 negative_class=True，加载负样本 prompt 并拼接 ------------
        if self.negative_class:
            negative_prompt_dict = {}   # {classname: [(negativeclass, description), ...] }
            for classname in classnames:
                hard_negative_classes = text_encoder_prompts[classname]["Hard Negatives"]
                negative_class_list = []
                for hard_negative_class in hard_negative_classes:
                    hard_name = hard_negative_class["NegativeClassName"]
                    hard_prompts = hard_negative_class["Negative Prompt"]
                    negative_class_list.append((hard_name, hard_prompts))
                
                negative_prompt_dict[classname] = negative_class_list

            print("[BERTPromptLearner] negative_class=True: loading negative prompts for each class.")
            self.neg_cls_count = n_cls  # in this version, each has one negative prompt
            
            neg_embeddings = []
            for i, cname in enumerate(classnames):
                neg_class = negative_prompt_dict[cname][0]  # only add one negative sample
                # form prompts by "<classname>: <desc>"
                neg_prompt = f'{neg_class[0]}: {neg_class[1]}'
                # tokenize & embedding
                tokenized_neg = tokenizer([neg_prompt], return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    # shape=(1, seq_len, hidden_dim)
                    neg_emb = bert_model.embeddings(
                        input_ids=tokenized_neg["input_ids"].to(self.device)
                    ).detach()
                neg_embeddings.append(neg_emb)

            # shape=(n_cls, seq_len, hidden_dim)
            neg_embeddings = torch.cat(neg_embeddings, dim=0)
            base_embedding = torch.cat([base_embedding, neg_embeddings], dim=0)

        # prefix = base_embedding[:, :0, :] => empth
        token_prefix = base_embedding[:, :0, :]  # shape=(batch, 0, hidden_dim)
        # suffix = base_embedding[:, n_ctx:, :]
        token_suffix = base_embedding[:, self.n_ctx:, :]  # shape=(batch, seq_len - n_ctx, hidden_dim)

        self.register_buffer("token_prefix", token_prefix)
        self.register_buffer("token_suffix", token_suffix)
        self.register_buffer("base_embedding", base_embedding)

        self.ctx_global = None
        if n_ctx > 0:
            if CSC:
                print("[BERTPromptLearner] Using class-specific contexts.")
                if random_init:
                    print("Random initialization for each class.")
                    ctx_global = torch.empty(self.pos_cls_count, n_ctx, ctx_dim)
                    nn.init.normal_(ctx_global, std=0.02)
                else:
                    print("Initializing with token embeddings (class-specific).")
                    ctx_global = base_embedding[:self.pos_cls_count, :n_ctx, :].clone()
            else:
                print("[BERTPromptLearner] Using a generic (shared) context.")
                if random_init:
                    print("Random initialization (shared).")
                    ctx_global = torch.empty(n_ctx, ctx_dim)
                    nn.init.normal_(ctx_global, std=0.02)
                else:
                    print("Initializing with token embeddings (shared).")
                    tmp = base_embedding[:self.pos_cls_count, :n_ctx, :].mean(dim=0, keepdim=True)  # (1, n_ctx, dim)
                    ctx_global = tmp.squeeze(0)  # (n_ctx, dim)

            self.ctx_global = nn.Parameter(ctx_global)
        self.batch_size_total = self.pos_cls_count + self.neg_cls_count

    def forward(self):
            """
            返回 shape = (pos_cls_count+neg_cls_count, seq_len, hidden_dim).
            其中:
            - 对前 pos_cls_count 行(正样本): 用 self.ctx_global 替换掉 base_embedding 的前 n_ctx tokens
            - 对后 neg_cls_count 行(负样本): 保持原始 embedding (即等效于 prefix+suffix = entire sequence)
            """
            be = self.base_embedding  # (batch_size_total, seq_len, hidden_dim)
            n_ctx = self.n_ctx

            if self.ctx_global is None or n_ctx == 0:
                # 没有上下文 => 大家都是原 embedding
                return be

            # ---- 拆分出正/负样本 ----
            pos_part = be[: self.pos_cls_count]  # (pos_cls_count, seq_len, dim)
            neg_part = be[self.pos_cls_count :]  # (neg_cls_count, seq_len, dim)

            # ---- 构建 prompt_pos ----
            # prefix_pos = pos_part[:, :0, :] => shape=(pos_cls_count, 0, dim) (可以省略，直接不拼)
            # suffix_pos = pos_part[:, n_ctx:, :]
            # 中间插入 self.ctx_global
            # 1) 准备可学习上下文(正样本专用)
            if self.CSC:
                # shape=(pos_cls_count, n_ctx, dim)
                ctx_pos = self.ctx_global
            else:
                # shared => shape=(n_ctx, dim) => expand到 (pos_cls_count, n_ctx, dim)
                ctx_pos = self.ctx_global.unsqueeze(0).expand(self.pos_cls_count, -1, -1)
            prompt_pos = torch.cat([self.token_prefix, ctx_pos, self.token_suffix], dim=1)

            # suffix_pos = pos_part[:, n_ctx:, :]  # (pos_cls_count, seq_len - n_ctx, dim)
            # # 拼起来 => (pos_cls_count, n_ctx + (seq_len-n_ctx), dim) = (pos_cls_count, seq_len, dim)
            # prompt_pos = torch.cat([ctx_pos, suffix_pos], dim=1)

            # ---- 构建 prompt_neg ----
            # 对负样本不替换上下文 => 直接用原序列
            # 也可以理解为 prefix_neg = neg_part[:, :n_ctx, :], suffix_neg = neg_part[:, n_ctx:, :],
            # prompt_neg = prefix_neg + suffix_neg => 全序列
            prompt_neg = neg_part  # shape=(neg_cls_count, seq_len, dim)

            # 拼接到一起
            prompts = torch.cat([prompt_pos, prompt_neg], dim=0)  # (pos_cls_count+neg_cls_count, seq_len, dim)
            return prompts


class TextEncoder_server_bert(nn.Module):
    def __init__(self, classnames, n_ctx, pretrained_model_name="bert-base-uncased", CSC=True, random_init=True, manual_prompt=True, negative_class=False, dataset=None):
        super().__init__()

        # Use the updated BERTPromptLearner
        self.prompt_learner = BERTPromptLearner(n_ctx, classnames, pretrained_model_name, CSC, random_init, manual_prompt=manual_prompt, negative_class=negative_class, dataset=dataset)
        self.bert_model = BertModel.from_pretrained(pretrained_model_name)
        # self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))  # Initial logit scale
        self.logit_scale = nn.Parameter(torch.tensor(4.6052))   # Initial logit scale is the same with the CLIP model
        self.num_classes = len(classnames)

    def get_text_prototypes(self):
        with torch.no_grad():
            # Obtain trainable prompts
            prompts = self.prompt_learner()
            # Forward pass through BERT to get the text features
            outputs = self.bert_model(inputs_embeds=prompts)
            text_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize
        return text_features.detach()

    def forward(self, global_vision_prototype):
        if isinstance(global_vision_prototype, defaultdict):
            # feature_list = list(global_vision_prototype.values())
            feature_list = [global_vision_prototype[key] for key in range(self.num_classes)]
            image_features = torch.stack(feature_list)
        else:
            image_features = global_vision_prototype
        # print(f'global_vision_prototype after convert:{image_features}')
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Obtain text features from prompts
        prompts = self.prompt_learner()
        outputs = self.bert_model(inputs_embeds=prompts)
        text_features = outputs.last_hidden_state[:, 0, :]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits