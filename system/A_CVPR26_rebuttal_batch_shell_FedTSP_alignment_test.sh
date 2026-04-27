# bash ABLATION_alignment_FedTSPv4_CVPR26_test_CLIP.sh Cifar100 100 dir 1.0 20 HtFE9 7 20 10 100 0 50 0 LLM_prompts 3 &
# bash ABLATION_alignment_FedTSPv4_CVPR26_test_BERT.sh Cifar100 100 dir 1.0 20 HtFE9 7 20 10 100 1 50 0 LLM_prompts 3 &

# bash ABLATION_alignment_FedTSPv4_CVPR26_test_CLIP.sh Cifar10 10 dir 1.0 20 HtFE9 7 1 1 100 2 50 0 LLM_prompts 3 &
# bash ABLATION_alignment_FedTSPv4_CVPR26_test_BERT.sh Cifar10 10 dir 1.0 20 HtFE9 7 1 1 100 3 50 0 LLM_prompts 3 &

# bash ABLATION_alignment_FedTSPv4_CVPR26_test_CLIP.sh Cifar100 100 dir 1.0 20 HtFE9 7 20 10 100 4 200 0 LLM_prompts 3 &
# bash ABLATION_alignment_FedTSPv4_CVPR26_test_BERT.sh Cifar100 100 dir 1.0 20 HtFE9 7 20 10 100 5 51 0 LLM_prompts 3 &

# bash ABLATION_alignment_FedTSPv4_CVPR26_test_CLIP.sh Cifar10 10 dir 1.0 20 HtFE9 7 1 1 100 6 200 0 LLM_prompts 3 &
# bash ABLATION_alignment_FedTSPv4_CVPR26_test_BERT.sh Cifar10 10 dir 1.0 20 HtFE9 7 1 1 100 7 51 0 LLM_prompts 3 &
# wait

bash ABLATION_alignment_FedTSPv4_CVPR26_test_CLIP.sh Cifar100 100 dir 1.0 20 HtFE9 7 20 10 100 0 200 0 LLM_prompts 3 &
bash ABLATION_alignment_FedTSPv4_CVPR26_test_CLIP.sh Cifar10 10 dir 1.0 20 HtFE9 7 1 1 100 1 200 0 LLM_prompts 3 &
bash ABLATION_alignment_FedTSPv4_CVPR26_test_BERT.sh Cifar100 100 dir 1.0 20 HtFE9 7 20 10 100 2 200 0 LLM_prompts 3 &
bash ABLATION_alignment_FedTSPv4_CVPR26_test_BERT.sh Cifar10 10 dir 1.0 20 HtFE9 7 1 1 100 3 200 0 LLM_prompts 3 &

bash ABLATION_alignment_FedTSPv4_CVPR26_test_CLIP.sh Cifar100 100 dir 0.1 20 HtFE9 7 20 10 100 4 200 0 LLM_prompts 3 &
bash ABLATION_alignment_FedTSPv4_CVPR26_test_CLIP.sh Cifar10 10 dir 0.1 20 HtFE9 7 1 1 100 5 200 0 LLM_prompts 3 &
bash ABLATION_alignment_FedTSPv4_CVPR26_test_BERT.sh Cifar100 100 dir 0.1 20 HtFE9 7 20 10 100 6 200 0 LLM_prompts 3 &
bash ABLATION_alignment_FedTSPv4_CVPR26_test_BERT.sh Cifar10 10 dir 0.1 20 HtFE9 7 1 1 100 7 200 0 LLM_prompts 3 &