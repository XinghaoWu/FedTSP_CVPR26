bash ABLATION_dynamic_FedTSPv4_CVPR26_CLIP.sh Cifar100 100 dir 0.1 20 HtFE9 7 20 10 100 0 100 0 LLM_prompts 3 0 0.5 50 &
bash ABLATION_dynamic_FedTGP_CVPR26_CLIP.sh Cifar100 100 dir 0.1 20 HtFE9 100 1 100 0.5 50 &
bash ABLATION_dynamic_FedProto_CVPR26_CLIP.sh Cifar100 100 dir 0.1 20 HtFE9 100 2 100 0.5 50 &

bash ABLATION_dynamic_FedTSPv4_CVPR26_CLIP.sh Cifar100 100 dir 0.5 20 HtFE9 7 20 10 100 3 100 0 LLM_prompts 3 0 1.0 50 &
bash ABLATION_dynamic_FedTGP_CVPR26_CLIP.sh Cifar100 100 dir 0.5 20 HtFE9 100 3 100 1.0 50 &
bash ABLATION_dynamic_FedProto_CVPR26_CLIP.sh Cifar100 100 dir 0.5 20 HtFE9 100 3 100 1.0 50 &

bash ABLATION_dynamic_FedTSPv4_CVPR26_CLIP.sh Cifar100 100 dir 0.1 20 HtFE9 7 20 10 100 4 100 0 LLM_prompts 3 0 1.0 50 &
bash ABLATION_dynamic_FedTGP_CVPR26_CLIP.sh Cifar100 100 dir 0.1 20 HtFE9 100 4 100 1.0 50 &
bash ABLATION_dynamic_FedProto_CVPR26_CLIP.sh Cifar100 100 dir 0.1 20 HtFE9 100 4 100 1.0 50 &

bash ABLATION_dynamic_FedTSPv4_CVPR26_CLIP.sh Cifar100 100 dir 0.5 20 HtFE9 7 20 10 100 5 100 0 LLM_prompts 3 0 0.1 50 &
bash ABLATION_dynamic_FedTGP_CVPR26_CLIP.sh Cifar100 100 dir 0.5 20 HtFE9 100 5 100 0.1 50 &
bash ABLATION_dynamic_FedProto_CVPR26_CLIP.sh Cifar100 100 dir 0.5 20 HtFE9 100 5 100 0.1 50 &

bash ABLATION_dynamic_FedTSPv4_CVPR26_CLIP.sh Cifar100 100 dir 1.0 20 HtFE9 7 20 10 100 6 100 0 LLM_prompts 3 0 0.5 50 &
bash ABLATION_dynamic_FedTGP_CVPR26_CLIP.sh Cifar100 100 dir 1.0 20 HtFE9 100 6 100 0.5 50 &
bash ABLATION_dynamic_FedProto_CVPR26_CLIP.sh Cifar100 100 dir 1.0 20 HtFE9 100 6 100 0.5 50 &

bash ABLATION_dynamic_FedTSPv4_CVPR26_CLIP.sh Cifar100 100 dir 1.0 20 HtFE9 7 20 10 100 7 100 0 LLM_prompts 3 0 0.1 50 &
bash ABLATION_dynamic_FedTGP_CVPR26_CLIP.sh Cifar100 100 dir 1.0 20 HtFE9 100 7 100 0.1 50 &
bash ABLATION_dynamic_FedProto_CVPR26_CLIP.sh Cifar100 100 dir 1.0 20 HtFE9 100 7 100 0.1 50 &
wait