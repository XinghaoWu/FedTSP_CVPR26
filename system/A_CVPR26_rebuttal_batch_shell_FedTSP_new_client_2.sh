bash ABLATION_newclients_FedTSPv4_CVPR26_CLIP.sh Cifar100 100 dir 0.5 20 HtFE9 7 20 10 100 0 200 0 LLM_prompts 3 1 16 100 &
bash ABLATION_newclients_FedTGP_CVPR26_CLIP.sh Cifar100 100 dir 0.5 20 HtFE9 100 1 200 1 16 100 &
bash ABLATION_newclients_FedProto_CVPR26_CLIP.sh Cifar100 100 dir 0.5 20 HtFE9 100 2 200 1 16 100 &
bash ABLATION_newclients_FedTSPv4_CVPR26_CLIP.sh Cifar100 100 dir 0.5 20 HtFE9 0 0 10 100 3 200 0 LLM_prompts 3 1 16 100 &


bash ABLATION_newclients_FedTSPv4_CVPR26_CLIP.sh Cifar100 100 dir 0.1 20 HtFE9 7 20 10 100 4 200 0 LLM_prompts 3 1 16 100 &
bash ABLATION_newclients_FedTGP_CVPR26_CLIP.sh Cifar100 100 dir 0.1 20 HtFE9 100 5 200 1 16 100 &
bash ABLATION_newclients_FedProto_CVPR26_CLIP.sh Cifar100 100 dir 0.1 20 HtFE9 100 6 200 1 16 100 &
bash ABLATION_newclients_FedTSPv4_CVPR26_CLIP.sh Cifar100 100 dir 0.1 20 HtFE9 0 0 10 100 7 200 0 LLM_prompts 3 1 16 100 &

wait