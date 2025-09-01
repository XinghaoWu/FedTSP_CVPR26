# bash ABLATION_Prompt_FedTSPv4_CLIP.sh Cifar10 10 dir 0.1 20 HtFE9 7 1 1 100 0 100 1 LLM_prompts_claude 3 &
# bash ABLATION_Prompt_FedTSPv4_CLIP.sh Cifar10 10 dir 0.1 20 HtFE9 7 1 1 100 0 100 1 LLM_prompts_gemini 3 &
# wait
# bash ABLATION_Prompt_FedTSPv4_CLIP.sh Cifar10 10 dir 0.5 20 HtFE9 7 1 1 100 0 100 1 LLM_prompts_claude 3 &
# bash ABLATION_Prompt_FedTSPv4_CLIP.sh Cifar10 10 dir 0.5 20 HtFE9 7 1 1 100 0 100 1 LLM_prompts_gemini 3 &
# wait
# bash ABLATION_Prompt_FedTSPv4_CLIP.sh Cifar10 10 dir 1.0 20 HtFE9 7 1 1 100 0 100 1 LLM_prompts_claude 3 &
# bash ABLATION_Prompt_FedTSPv4_CLIP.sh Cifar10 10 dir 1.0 20 HtFE9 7 1 1 100 0 100 1 LLM_prompts_gemini 3 &

# bash ABLATION_Prompt_FedTSPv4_CLIP.sh Cifar100 100 dir 0.1 20 HtFE9 7 20 10 100 0 100 1 LLM_prompts_claude 3 &
# bash ABLATION_Prompt_FedTSPv4_CLIP.sh Cifar100 100 dir 0.1 20 HtFE9 7 20 10 100 0 100 1 LLM_prompts_gemini 3 &
# wait
# bash ABLATION_Prompt_FedTSPv4_CLIP.sh Cifar100 100 dir 0.5 20 HtFE9 7 20 10 100 0 100 1 LLM_prompts_claude 3 &
# bash ABLATION_Prompt_FedTSPv4_CLIP.sh Cifar100 100 dir 0.5 20 HtFE9 7 20 10 100 0 100 1 LLM_prompts_gemini 3 &
# wait
# bash ABLATION_Prompt_FedTSPv4_CLIP.sh Cifar100 100 dir 1.0 20 HtFE9 7 20 10 100 0 100 1 LLM_prompts_claude 3 &
# bash ABLATION_Prompt_FedTSPv4_CLIP.sh Cifar100 100 dir 1.0 20 HtFE9 7 20 10 100 0 100 1 LLM_prompts_gemini 3 &

# bash ABLATION_Prompt_FedTSPv4_CLIP.sh Cifar10 10 dir 0.1 20 HtFE9 7 1 1 100 0 100 0 LLM_prompts_llama3_8b_instruct 3 &
# bash ABLATION_Prompt_FedTSPv4_CLIP.sh Cifar10 10 dir 0.5 20 HtFE9 7 1 1 100 0 100 0 LLM_prompts_llama3_8b_instruct 3 &
# bash ABLATION_Prompt_FedTSPv4_CLIP.sh Cifar10 10 dir 1.0 20 HtFE9 7 1 1 100 0 100 0 LLM_prompts_llama3_8b_instruct 3 &

bash ABLATION_Prompt_FedTSPv4_CLIP.sh Cifar100 100 dir 0.1 20 HtFE9 7 20 10 100 0 100 0 LLM_prompts_llama3_8b_instruct 3 &
bash ABLATION_Prompt_FedTSPv4_CLIP.sh Cifar100 100 dir 0.5 20 HtFE9 7 20 10 100 0 100 0 LLM_prompts_llama3_8b_instruct 3 &
bash ABLATION_Prompt_FedTSPv4_CLIP.sh Cifar100 100 dir 1.0 20 HtFE9 7 20 10 100 0 100 0 LLM_prompts_llama3_8b_instruct 3 &