import arguments
import ar_maml


if __name__ == '__main__':

    args = arguments.parse_args()
    exp_string_armaml_outer = 'armaml_' + 'model_outer_' + args.init_dis + '_tpm_' + str(args.tasks_per_metaupdate) + '_K' + str(args.update_batch_size) + '_n' + str(args.n_iter) + args.flow_type + '_dis_interval' + str(args.dist_interval) + '_lr' + str(args.lr_dist) + '.pt'
    exp_string_armaml_dist = 'armaml_' + 'dist_model_' + args.init_dis + '_tpm_' + str(args.tasks_per_metaupdate) + '_K' + str(args.update_batch_size) + '_n' + str(args.n_iter) + args.flow_type + '_dis_interval' + str(args.dist_interval) + '_lr' + str(args.lr_dist) + '.pt'
    
    
    if args.ar_maml and args.train:
        print('armaml, train', args.init_dis)
        ar_maml.ar_ml_run(args, log_interval=args.log_interval, dist_interval=args.dist_interval, rerun=True, game_framework=True)
    elif args.ar_maml and not args.train:
        print(args.train)
        print('armaml, test')
        ar_maml.eval(args, exp_string_armaml_dist, exp_string_armaml_outer)
    
