# bash ABLATION_Losstype_FedTSPv4_CLIP.sh Cifar10 10 dir 0.1 20 HtFE9 7 1 1 100 0 3 1 LLM_prompts 3 0 L2
# sleep 8h
# bash ABLATION_Losstype_FedTSPv4_CLIP.sh Cifar10 10 dir 0.1 20 HtFE9 7 1 1 100 0 100 1 LLM_prompts 3 0 L2 &
# bash ABLATION_Losstype_FedTSPv4_CLIP.sh Cifar10 10 dir 0.5 20 HtFE9 7 1 1 100 0 100 1 LLM_prompts 3 0 L2 &
# bash ABLATION_Losstype_FedTSPv4_CLIP.sh Cifar10 10 dir 1.0 20 HtFE9 7 1 1 100 0 100 1 LLM_prompts 3 0 L2 &
# wait
bash ABLATION_Losstype_FedTSPv4_CLIP.sh Cifar100 100 dir 0.1 20 HtFE9 7 20 10 100 0 100 1 LLM_prompts 3 0 L2 &
bash ABLATION_Losstype_FedTSPv4_CLIP.sh Cifar100 100 dir 0.5 20 HtFE9 7 20 10 100 0 100 1 LLM_prompts 3 0 L2 &
bash ABLATION_Losstype_FedTSPv4_CLIP.sh Cifar100 100 dir 1.0 20 HtFE9 7 20 10 100 0 100 1 LLM_prompts 3 0 L2 &