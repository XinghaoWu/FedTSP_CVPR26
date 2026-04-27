bash ABLATION_alignment_FedTSPv4_CVPR26_CLIP.sh Cifar100 100 dir 1.0 20 HtFE9 7 20 10 100 0 200 0 LLM_prompts 3 &
bash ABLATION_alignment_FedTSPv4_CVPR26_BERT.sh Cifar100 100 dir 1.0 20 HtFE9 7 20 10 100 1 200 0 LLM_prompts 3 &

bash ABLATION_alignment_FedTSPv4_CVPR26_CLIP.sh Cifar10 10 dir 1.0 20 HtFE9 7 1 1 100 2 200 0 LLM_prompts 3 &
bash ABLATION_alignment_FedTSPv4_CVPR26_BERT.sh Cifar10 10 dir 1.0 20 HtFE9 7 1 1 100 3 200 0 LLM_prompts 3 &
wait