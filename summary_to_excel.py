import argparse

import re
import openpyxl

def get_arguments():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar100_dir_1.0_balance_20")
    parser.add_argument('-nb', "--num_classes", type=int, default=100)
    parser.add_argument('-m', "--model_family", type=str, default="HtFE2")
    parser.add_argument('-lbs', "--batch_size", type=int, default=100)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=5)
    parser.add_argument('-ls', "--local_epochs", type=int, default=5,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedTSPv2")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='temp')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=98635)
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")

    parser.add_argument("--seed", type=int, default=0)



    # FedProto/ours/FedDistill (gamma)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=32)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.1)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=100)
    # FML
    parser.add_argument('-al', "--alpha", type=float, default=0.5)
    parser.add_argument('-bt', "--beta", type=float, default=0.5)
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.01)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.95)
    # FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=0.01)
    # FedTGP
    parser.add_argument('-mart', "--margin_threthold", type=float, default=100.0)
    # FedKTL
    parser.add_argument('-GPath', "--generator_path", type=str, default='stylegan/stylegan-xl-models/imagenet64.pkl')
    parser.add_argument('-prompt', "--stable_diffusion_prompt", type=str, default='a cat')
    parser.add_argument('-sbs', "--server_batch_size", type=int, default=100)
    parser.add_argument('-gbs', "--gen_batch_size", type=int, default=4,
                        help="Not related to the performance. A small value saves GPU memory.")
    parser.add_argument('-mu', "--mu", type=float, default=50.0)

    # ours
    parser.add_argument('--len_prompt', default=24, type=int, help='the length of prompts') # v2
    parser.add_argument('--p_classifier', type=int, default=1, help='whether to personalize classifier')    # v2
    parser.add_argument('--p_prompt', type=int, default=0, help='whether to personalize prompt')
    parser.add_argument('--alter', type=int, default=0, help='whether to use alternate training')
    parser.add_argument('--update_prompt', default=True, action='store_false', help='whether to update trainable prompt')
    parser.add_argument('--prompt_epoch', type=int, default=1, help='the number of training prompt epochs, only useful when --alter=1')   # v2: server update round
    parser.add_argument('--prompt_lr', type=float, default=0.01, help='learning rate for prompt')   # v2: server learning rate
    parser.add_argument('--CSC', default=True, action='store_false', help='whether use class-specific prompt')  # v2
    parser.add_argument('--vision_proto', type=float, default=0, help='whether to align with the vision prototype, set to 0 to disable')    # v2

    # FedTSPv2
    parser.add_argument('--EMA_alpha', type=float, default=0, help='EMA ratio')
    parser.add_argument('--prompt_EMA_alpha', type=float, default=0, help='prompt EMA ratio')
    parser.add_argument('--prompt_random_init', default=True, action='store_false', help='whether to randomly initialize prompt')
    parser.add_argument('--server_training_freq', type=int, default=1, help='server training freq')

    # summary arguments
    parser.add_argument('--summary_date', type=str, default='10.23', help='summary file name')

    args = parser.parse_args()
    return args





def parse_summary(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content into individual experiments
    experiments = content.split('==================================================')
    experiments = [exp.strip() for exp in experiments if exp.strip()]

    parsed_data = []

    for experiment in experiments:
        exp_data = {}

        # Use regular expressions to extract hyperparameters and results
        hyperparams = re.findall(r'(\w+) : ([\d\.\-e]+|True|False)', experiment)
        for param, value in hyperparams:
            exp_data[param] = value

        # Extract results
        accuracy_match = re.search(r'Best accuracy : ([\d\.]+)', experiment)
        if accuracy_match:
            exp_data['Best accuracy'] = float(accuracy_match.group(1))

        epoch_match = re.search(r'Best epoch : (\d+)', experiment)
        if epoch_match:
            exp_data['Best epoch'] = int(epoch_match.group(1))

        parsed_data.append(exp_data)

    return parsed_data


def write_to_excel(data, output_path):
    # Create a new Excel workbook and select the active worksheet
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = 'Experiment Results'

    # Define the header
    headers = list(data[0].keys())
    sheet.append(headers)

    # Write data to the sheet
    for exp_data in data:
        row = [exp_data.get(header, None) for header in headers]
        sheet.append(row)

    # Save the workbook to a file
    workbook.save(output_path)


if __name__ == '__main__':
    args = get_arguments()
    # Parse the summary file

    # experiment result dir
    dir_path = f'./logs/{args.dataset}/{args.model_family}/{args.algorithm}/gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}/'

    summary_file = f'summary_{args.summary_date}.txt'
    parsed_data = parse_summary(dir_path + summary_file)

    # Write the parsed data to an Excel file
    output_excel = dir_path+f'{args.dataset}_{args.model_family}_{args.algorithm}_gr{args.global_rounds}_ep{args.local_epochs}_bs{args.batch_size}_nc{args.num_clients}_experiment_results_{args.summary_date}.xlsx'
    write_to_excel(parsed_data, output_excel)

    print(f"Data has been written to {output_excel}")



