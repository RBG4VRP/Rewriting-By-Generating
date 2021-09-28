from region_vrp import *
import time


if __name__ == '__main__':
    args=parse()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.eval_data is None:
        filename='data/data_{}_100_50.pkl'.format(args.size)
    else:
        filename=args.eval_data
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    
    if filename[-3:]=='vrp':
        print('vrp file')

    device = torch.device('cuda:0')
    args.device=device
    problem = load_problem('cvrp')
    model = AttentionModel(
        128,
        128,
        problem,
        n_encode_layers=3,
        mask_inner=True,
        mask_logits=True,
        normalization='batch',
        tanh_clipping=10,
        checkpoint_encoder=False,
        shrink_size=None
    ).to(device)
    model.set_decode_type("greedy", temp=1)
    model.eval()
    selectmodel = SelectModel(args=args, device=device).to(device)
    
    with open('{}'.format(args.load_model),'rb') as f:
        state_dicts=pickle.load(f)
        model.load_state_dict(state_dicts[0])
        selectmodel.load_state_dict(state_dicts[1])

    t1=time.time()
    print('time1=',t1)
    costs, _ = eval(dataset, model, selectmodel, args)
    t2=time.time()

    print('Problem size=',args.size,', Scaled Mean cost=',costs.mean()*args.eval_scale_factor, ', Mean time=',(t2-t1)/len(dataset))