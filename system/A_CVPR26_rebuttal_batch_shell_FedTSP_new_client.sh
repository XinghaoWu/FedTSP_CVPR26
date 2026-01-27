bash ABLATION_newclients_FedTSPv4_CVPR26_CLIP.sh Cifar100 100 dir 0.5 20 HtFE9 7 20 10 100 0 150 0 LLM_prompts 3 1 16 50 &
bash ABLATION_newclients_FedTGP_CVPR26_CLIP.sh Cifar100 100 dir 0.5 20 HtFE9 100 1 150 1 16 50 &
bash ABLATION_newclients_FedProto_CVPR26_CLIP.sh Cifar100 100 dir 0.5 20 HtFE9 100 2 150 1 16 50 &
bash ABLATION_newclients_FedTSPv4_CVPR26_CLIP.sh Cifar100 100 dir 0.5 20 HtFE9 0 0 10 100 3 150 0 LLM_prompts 3 1 16 50 &
wait