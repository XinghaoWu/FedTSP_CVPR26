bash ABLATION_dynamic_FedTSPv4_CVPR26_CLIP.sh Cifar100 100 dir 0.1 20 HtFE9 7 20 10 100 0 200 0 LLM_prompts 3 0 0.5 100 &
bash ABLATION_dynamic_FedTGP_CVPR26_CLIP.sh Cifar100 100 dir 0.1 20 HtFE9 100 1 200 0.5 100 &
bash ABLATION_dynamic_FedProto_CVPR26_CLIP.sh Cifar100 100 dir 0.1 20 HtFE9 100 2 200 0.5 100 &
wait