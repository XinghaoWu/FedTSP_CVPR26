from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch

from collections import defaultdict


# class BERTPromptLearner(nn.Module):
#     def __init__(self, n_ctx, classnames, pretrained_model_name="bert-base-uncased"):
#         super().__init__()
#         self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
#         self.bert_model = BertModel.from_pretrained(pretrained_model_name)
#
#         self.n_cls = len(classnames)
#         self.n_ctx = n_ctx  # Number of context tokens
#         self.ctx_embeddings = nn.Parameter(torch.randn(self.n_cls, n_ctx, self.bert_model.config.hidden_size))
#
#         # Prepare class-specific prompts
#         self.prompts = [f"A photo of a {name}." for name in classnames]
#
#     def forward(self):
#         # Tokenize prompts to get fixed token embeddings (non-trainable)
#         tokenized_prompts = self.tokenizer(self.prompts, return_tensors="pt", padding=True, truncation=True)
#         with torch.no_grad():
#             fixed_embeddings = self.bert_model.embeddings(input_ids=tokenized_prompts["input_ids"]).detach()
#
#         # Insert trainable context tokens in each prompt
#         cls_embeddings = torch.cat([self.ctx_embeddings, fixed_embeddings[:, self.n_ctx:, :]], dim=1)
#         return cls_embeddings

class BERTPromptLearner(nn.Module):
    def __init__(self, n_ctx, classnames, pretrained_model_name="bert-base-uncased", CSC=True, random_init=True):
        super().__init__()

        # Initialize tokenizer and model temporarily to obtain fixed suffix embeddings
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        bert_model = BertModel.from_pretrained(pretrained_model_name)

        # Define variables based on input arguments
        n_cls = len(classnames)
        ctx_dim = bert_model.config.hidden_size
        self.n_ctx = n_ctx
        self.CSC = CSC  # Class-Specific Contexts

        # Prompt template setup
        prompt_prefix = "A photo of a"
        prompts = [f"{prompt_prefix} {name}." for name in classnames]

        # Tokenize prompts to get non-trainable suffix embeddings
        tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            fixed_embeddings = bert_model.embeddings(input_ids=tokenized_prompts["input_ids"]).detach()

        # Initialize ctx_global based on n_ctx and CSC options
        if n_ctx > 0:
            if CSC:
                # Class-Specific Contexts: Each class has its own prompt embeddings
                print("Initializing class-specific contexts")
                if random_init:
                    print("Random initialization")
                    ctx_global = torch.empty(n_cls, n_ctx, ctx_dim)
                    nn.init.normal_(ctx_global, std=0.02)
                else:
                    print("Initializing with token embeddings")
                    ctx_global = fixed_embeddings[:, :n_ctx, :].clone()
            else:
                # Shared Contexts: All classes share the same prompt embeddings
                print("Initializing a generic context")
                if random_init:
                    print("Random initialization")
                    ctx_global = torch.empty(n_ctx, ctx_dim)
                    nn.init.normal_(ctx_global, std=0.02)
                else:
                    print("Initializing with token embeddings")
                    ctx_global = fixed_embeddings[:, :n_ctx, :].mean(dim=0, keepdim=True).clone()

            # Convert ctx_global to a learnable parameter
            self.ctx_global = nn.Parameter(ctx_global)
        else:
            self.ctx_global = None  # If no context is needed, set to None

        # Save non-trainable prefix and suffix embeddings as buffers
        self.register_buffer("token_prefix", fixed_embeddings[:, :0, :])  # Empty prefix (0 tokens)
        self.register_buffer("token_suffix", fixed_embeddings[:, n_ctx:, :])  # Fixed suffix (from n_ctx to end)

    def forward(self):
        # Concatenate the trainable context (ctx_global) at the beginning, followed by the fixed suffix
        prefix = self.token_prefix  # Empty if n_ctx > 0
        suffix = self.token_suffix

        if self.ctx_global is not None:
            ctx = self.ctx_global
            if ctx.dim() == 2:  # Shared context
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            prompts = torch.cat([prefix, ctx, suffix], dim=1)  # Insert ctx at position 0
        else:
            prompts = torch.cat([prefix, suffix], dim=1)

        return prompts


class TextEncoder_server_bert(nn.Module):
    def __init__(self, classnames, n_ctx, pretrained_model_name="bert-base-uncased", CSC=True, random_init=True):
        super().__init__()

        # Use the updated BERTPromptLearner
        self.prompt_learner = BERTPromptLearner(n_ctx, classnames, pretrained_model_name, CSC, random_init)
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

# class TextEncoder_server(nn.Module):
#     def __init__(self, classnames, n_ctx, pretrained_model_name="bert-base-uncased"):
#         super().__init__()
#         # Use BERT-based prompt learner
#         self.prompt_learner = BERTPromptLearner(n_ctx, classnames, pretrained_model_name)
#         self.bert_model = BertModel.from_pretrained(pretrained_model_name)
#         self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))  # Initialize scale
#
#     def get_text_prototypes(self):
#         with torch.no_grad():
#             # Obtain trainable prompts
#             prompts = self.prompt_learner()
#             # Forward pass through BERT to get the text features
#             outputs = self.bert_model(inputs_embeds=prompts)
#             text_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
#             text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize
#         return text_features.detach()
#
#     def forward(self, global_vision_prototype):
#         image_features = global_vision_prototype / global_vision_prototype.norm(dim=-1, keepdim=True)
#
#         # Obtain text features from prompts
#         prompts = self.prompt_learner()
#         outputs = self.bert_model(inputs_embeds=prompts)
#         text_features = outputs.last_hidden_state[:, 0, :]
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#
#         # Compute logits
#         logit_scale = self.logit_scale.exp()
#         logits = logit_scale * image_features @ text_features.t()
#         return logits