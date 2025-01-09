import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Fast Context Adaptation via Meta-Learning (CAVIA),'
                                                 'Regression experiments')

    parser.add_argument('--task', type=str, default='sine', help='problem setting: sine or celeba or acrobot or pendulum')
    parser.add_argument('--run_type', type=str, default='armaml', help='armaml, maml, drmaml')
    parser.add_argument('--entropy_weight', type=float, default=0.1, help='entropy weight to constrain the distributional shifts')

    parser.add_argument('--n_iter', type=int, default=50001, help='number of meta-iterations')

    parser.add_argument('--tasks_per_metaupdate', type=int, default=25, help='the number of tasks in each batch')# 16 for acrobot, 25 for sinusoid
    parser.add_argument('--task_num_data_points', type=int, default=10, help='number of few-shot data points')#200 for acrobot, 10 for sinusoid
    parser.add_argument('--update_batch_size', type=int, default=5, help='number of few-shot data points')# 10 for acrobot, 5 for sinusoid

    parser.add_argument('--k_meta_train', type=int, default=15, help='data points in task training set (during meta training, inner loop)')
    parser.add_argument('--k_meta_test', type=int, default=15, help='data points in task test set (during meta training, outer loop)')
    parser.add_argument('--k_shot_eval', type=int, default=15, help='data points in task training set (during evaluation)')

    parser.add_argument('--lr_inner', type=float, default=0.001, help='inner-loop learning rate (task-specific)')
    parser.add_argument('--lr_meta', type=float, default=0.001, help='outer-loop learning rate')
    parser.add_argument('--lr_dist', type=float, default=0.0001, help='Distribution_Adversary learning rate')


    parser.add_argument('--num_inner_updates', type=int, default=1, help='number of inner-loop updates (during training)')

    parser.add_argument('--num_context_params', type=int, default=5, help='number of context parameters (added at first layer)')
    parser.add_argument('--num_hidden_layers', type=int, nargs='+', default=[128, 128, 128])

    parser.add_argument('--first_order', action='store_true', default=False, help='run first-order version')

    parser.add_argument('--ar_maml', action='store_true', default=False, help='run MAML')
    parser.add_argument('--train', action='store_true', default=False)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_number', type=int, default=255)
    parser.add_argument('--dist_interval', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=100)

    # commands specific to the CelebA image completion task
    parser.add_argument('--use_ordered_pixels', action='store_true', default=False)

    parser.add_argument('--latent_size', type=int, default=2, help='latent size in nf flows')
    parser.add_argument('--num_latent_layers', type=int, default=2, help='number of latent layers)')
    parser.add_argument('--flow_type', type=str, default='Planar_Flow', help='Planar_Flow or Radial_Flow')
    parser.add_argument('--init_dis', type=str, default='Uniform', help='Uniform or Normal')

    parser.add_argument('--transformed_dis', action='store_true', default=False)
    
    parser.add_argument('--confidence_level', type=float, default=0.5)
    parser.add_argument('--step_size', type=float, default=0.01)


    # parser.add_argument('--device', type=int, default=1, help='number of inner-loop updates (during training)')
    args = parser.parse_args()
    # args = parser.parse_args(args=[])

    # use the GPU if available
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args
