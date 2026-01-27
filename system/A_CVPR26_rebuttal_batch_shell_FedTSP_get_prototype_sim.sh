# bash ABLATION_get_prototype_sim_FedTSP_CLIP.sh Cifar10 10 dir 1.0 20 HtFE9 7 1 1 100 0 100 LLM_prompts 3 get_proto_sim cosine &
# bash ABLATION_get_prototype_sim_FedTSP_CLIP.sh Cifar100 100 dir 1.0 20 HtFE9 7 20 10 100 0 100 LLM_prompts 3 get_proto_sim cosine &

# wait

# bash ABLATION_get_prototype_sim_FedTSP_ablation_CLIP.sh Cifar10 10 dir 1.0 20 HtFE9 7 1 1 100 0 100 LLM_prompts_claude 3 get_proto_sim cosine &
# bash ABLATION_get_prototype_sim_FedTSP_ablation_CLIP.sh Cifar100 100 dir 1.0 20 HtFE9 7 20 10 100 0 100 LLM_prompts_claude 3 get_proto_sim cosine &

# wait

bash ABLATION_get_prototype_sim_FedTSP_ablation_CLIP.sh Cifar10 10 dir 1.0 20 HtFE9 7 1 1 100 0 100 LLM_prompts_gemini 3 get_proto_sim cosine &
bash ABLATION_get_prototype_sim_FedTSP_ablation_CLIP.sh Cifar100 100 dir 1.0 20 HtFE9 7 20 10 100 0 100 LLM_prompts_gemini 3 get_proto_sim cosine &

wait