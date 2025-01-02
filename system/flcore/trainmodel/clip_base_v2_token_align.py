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
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner_client(nn.Module):
    def __init__(self, n_ctx_num, classnames, clip_model, CSC=True, random_init=True, manual_prompt=True, negative_class=False, dataset=None):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = n_ctx_num  # the number of prompts
        ctx_init = ''
        dtype = clip_model.dtype
        device = next(clip_model.parameters()).device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.manual_prompt = manual_prompt
        self.negative_class = negative_class
        CSC = CSC

        if self.manual_prompt:
            prompt_prefix = 'A photo of a'

            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]

            print(f'[PromptLearner_client] manual_prompt=True: using prompts like "{prompt_prefix} <classname>."')

            tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
            with torch.no_grad():
                # embedding = clip_model.token_embedding(tokenized_prompts.cuda()).type(dtype)
                embedding = clip_model.token_embedding(tokenized_prompts.to(device)).type(dtype)

            # save for using in the following
            self.register_buffer("tokenized_prompts", tokenized_prompts)
            self.register_buffer("base_embedding", embedding)

        else:
            # manual_prompt=False:
            # Obtain 3 fine-grained descriptions for each class, concatenate them as “classname: description”
            # Compute the embeddings separately and then take the average

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

            print("[PromptLearner_client] manual_prompt=False: using fine-grained descriptions and averaging embeddings.")
            

            classnames_processed = [c.replace("_", " ") for c in classnames]

            # used to store the final embedding of each class
            final_embeddings = []
            # store tokenized version is used
            all_tokenized_prompts = []

            for i, cname in enumerate(classnames):
                c_proc = classnames_processed[i]
                # get fine grained description from fine_grained_dict
                desc_list = fine_grained_dict[cname]  # ["desc1", "desc2", "desc3"]
                # form prompts by "<classname>: <desc>"
                prompts_for_this_class = [f"{c_proc}: {desc}" for desc in desc_list]

                # tokenized
                tokenized = torch.cat([clip.tokenize(p) for p in prompts_for_this_class])
                with torch.no_grad():
                    emb_3 = clip_model.token_embedding(tokenized.to(device)).type(dtype)  # shape=(3, seqlen, dim)

                # obtrain mean embedding
                emb_mean = emb_3.mean(dim=0, keepdim=True)  # shape=(1, seqlen, dim)

                final_embeddings.append(emb_mean)
                all_tokenized_prompts.append(tokenized)

            # concat embeddings of all classes => shape = (n_cls, seqlen, dim)
            embedding = torch.cat(final_embeddings, dim=0)
            self.register_buffer("base_embedding", embedding)

            # record tokenized version
            self.register_buffer("tokenized_prompts", torch.cat(all_tokenized_prompts, dim=0))

        # 2) If negative_class=True, load the negative prompt embedding for each class
        # and append it to the batch dimension of the embedding (dim=0).
        # Note: For the embeddings of negative_class, do not concatenate self.ctx_global or prepend “A photo of a”.

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


            print("[PromptLearner_client] negative_class=True: loading negative prompts for each class.")

            negative_embeddings = []
            for i, cname in enumerate(classnames):
                neg_class = negative_prompt_dict[cname][0]  # only add one negative sample
                # form prompts by "<classname>: <desc>"
                neg_prompt = f'{neg_class[0]}: {neg_class[1]}'
                
                # Tokenize & embedding
                tokenized_neg = clip.tokenize(neg_prompt)
                with torch.no_grad():
                    neg_emb = clip_model.token_embedding(tokenized_neg.to(device)).type(dtype)
                negative_embeddings.append(neg_emb)
            
            # Append to the batch dimension of the original embedding
            # base_embedding shape: (n_cls, seqlen, dim)
            # First, stack the negative embeddings into the shape (n_cls, seqlen, dim),
            # then concatenate along dim=0.
            negative_embeddings = torch.cat(negative_embeddings, dim=0)  # (n_cls, seqlen, dim)
            
            # final embeddings = normal (n_cls, seqlen, dim) + negative (n_cls, seqlen, dim) => (2*n_cls, seqlen, dim)
            embedding_cat = torch.cat([self.base_embedding, negative_embeddings], dim=0)
            
            # restore to self.base_embedding
            self.register_buffer("base_embedding", embedding_cat)

            self.pos_cls_count = n_cls
            self.neg_cls_count = n_cls
        else:
            self.pos_cls_count = n_cls
            self.neg_cls_count = 0

        # 3) initialize learnable prompt（self.ctx_global）if n_ctx>0
        # -------------------------------------------------------------
        if n_ctx > 0:
            if CSC:
                print("[PromptLearner_client] Using class-specific contexts (CSC).")
                if random_init:
                    print("Random initialization of context vectors.")
                    ctx_vectors = torch.empty(self.pos_cls_count, n_ctx, ctx_dim, dtype=dtype)
                    nn.init.normal_(ctx_vectors, std=0.02)
                    self.ctx_global = nn.Parameter(ctx_vectors)  # (n_cls, n_ctx, dim)
                else:
                    print("Initializing with token embeddings (class-specific).")
                    self.ctx_global = nn.Parameter(
                        self.base_embedding[:self.pos_cls_count, 1 : 1 + n_ctx, :].clone()
                    )
            else:
                print("[PromptLearner_client] Using a generic (shared) context.")
                if random_init:
                    print("Random initialization of context vectors (shared).")
                    ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                    nn.init.normal_(ctx_vectors, std=0.02)
                    self.ctx_global = nn.Parameter(ctx_vectors)  # (n_ctx, dim)
                else:
                    print("Initializing with token embeddings (shared).")
                    # shape: (1, n_ctx, dim)
                    tmp = self.base_embedding[:self.pos_cls_count, 1 : 1 + n_ctx, :].mean(dim=0, keepdim=True)
                    self.ctx_global = nn.Parameter(tmp)
        else:
            self.ctx_global = None


        # print(embedding.shape)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        # self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        # self.name_lens = name_lens
        # self.class_token_position = "end"

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


class TextEncoder_server(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx_num=16, CSC=True, random_init=True, manual_prompt=True, negative_class=False, dataset=None):
        super().__init__()
        self.prompt_learner = PromptLearner_client(n_ctx_num, classnames, clip_model, CSC, random_init, manual_prompt=manual_prompt, negative_class=negative_class, dataset=dataset)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.num_classes = len(classnames)

    def get_text_prototypes(self):
        with torch.no_grad():
            tokenized_prompts = self.tokenized_prompts
            prompts = self.prompt_learner()

            text_features = self.text_encoder(prompts, tokenized_prompts)

        return text_features.detach()

    def forward(self, global_vision_prototype):
        # Check if global_vision_prototype is a defaultdict, and convert it to a tensor if necessary
        # print(f'global_vision_prototype before convert:{global_vision_prototype}')
        if isinstance(global_vision_prototype, defaultdict):
            # feature_list = list(global_vision_prototype.values())
            feature_list = [global_vision_prototype[key] for key in range(self.num_classes)]
            image_features = torch.stack(feature_list)
        else:
            image_features = global_vision_prototype
        # print(f'global_vision_prototype after convert:{image_features}')
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner()

        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits