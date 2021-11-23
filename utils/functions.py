import warnings

import torch
import numpy as np
import os
import json
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import torch.nn.functional as F


def load_problem(name):
    from problems import TSP, CVRP, SDVRP, OP, PCTSPDet, PCTSPStoch
    problem = {
        'tsp': TSP,
        'cvrp': CVRP,
        'sdvrp': SDVRP,
        'op': OP,
        'pctsp_det': PCTSPDet,
        'pctsp_stoch': PCTSPStoch,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU


def move_to(var, device):
    #print(var)
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    if isinstance(var, list):
        return [move_to(k, device) for k in var]
    if isinstance(var, tuple):
        return tuple([move_to(k, device) for k in var])
    return var.to(device)


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print('  [*] Loading model from {}'.format(load_path))

    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage)

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()

    state_dict.update(load_model_state_dict)

    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    return args


def load_model(path, epoch=None):
    from nets.attention_model import AttentionModel
    from nets.pointer_network import PointerNetwork

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, 'args.json'))

    problem = load_problem(args['problem'])

    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork
    }.get(args.get('model', 'attention'), None)
    assert model_class is not None, "Unknown model: {}".format(model_class)

    model = model_class(
        args['embedding_dim'],
        args['hidden_dim'],
        problem,
        n_encode_layers=args['n_encode_layers'],
        mask_inner=True,
        mask_logits=True,
        normalization=args['normalization'],
        tanh_clipping=args['tanh_clipping'],
        checkpoint_encoder=args.get('checkpoint_encoder', False),
        shrink_size=args.get('shrink_size', None)
    )
    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})

    model, *_ = _load_model_file(model_filename, model)

    model.eval()  # Put in eval mode

    return model, args


def parse_softmax_temperature(raw_temp):
    # Load from file
    if os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)


def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


def sample_many(inner_func, get_cost_func, input, batch_rep=1, iter_rep=1):
    """
    :param input: (batch_size, graph_size, node_dim) input node features
    :return:
    """
    input = do_batch_rep(input, batch_rep)

    costs = []
    pis = []
    for i in range(iter_rep):
        _log_p, pi = inner_func(input)
        # pi.view(-1, batch_rep, pi.size(-1))
        cost, mask = get_cost_func(input, pi)

        costs.append(cost.view(batch_rep, -1).t())
        pis.append(pi.view(batch_rep, -1, pi.size(-1)).transpose(0, 1))

    max_length = max(pi.size(-1) for pi in pis)
    # (batch_size * batch_rep, iter_rep, max_length) => (batch_size, batch_rep * iter_rep, max_length)
    pis = torch.cat(
        [F.pad(pi, (0, max_length - pi.size(-1))) for pi in pis],
        1
    )  # .view(embeddings.size(0), batch_rep * iter_rep, max_length)
    costs = torch.cat(costs, 1)

    # (batch_size)
    mincosts, argmincosts = costs.min(-1)
    # (batch_size, minlength)
    minpis = pis[torch.arange(pis.size(0), out=argmincosts.new()), argmincosts]

    return minpis, mincosts



def distance1(x,y,beta=0,depot_loc=torch.tensor([0.5,0.5]),eps1=0.00001,device=torch.device('cpu')):
#     print(x.shape,y.shape)
    dis=torch.sum((x-y)**2,-1).to(device)
    if beta!=0:
        a1=torch.sum((x-depot_loc)*(y-depot_loc),-1)
        b1=(torch.sqrt(torch.sum((x-depot_loc)**2,-1)+eps1))
        c1=(torch.sqrt(torch.sum((y-depot_loc)**2,-1)+eps1))
        ang=torch.acos(a1/b1/c1)
    else:
        ang=0
    return dis+beta*ang

#聚类,现在先假设loc,demand的shape是size*2,size
def K_means(K,loc,demand,beta=0,depot=torch.tensor([0.5,0.5]),step=10,plot=False,device=torch.device('cpu')):
    loc_center=torch.rand(K,2).to(device)

    for n in range(step):
        loc2_center=torch.zeros((K,2)).to(device)
        num_center=torch.zeros(K).to(device)
        track_loc=[]  #points for each center, None for zero; shape=[size1*2,size2*2,...,sizeK*2]
        track_demand=[]  #shape: size1*n
        for i in range(K):
            track_loc.append([])
            track_demand.append([])

        dis1=torch.argmin(distance1(loc[:,None,:],loc_center,beta,device=device),1)
        for i,j in enumerate(dis1):
            loc2_center[j]=loc2_center[j]+loc[i]
            num_center[j]=num_center[j]+1
            track_loc[j].append(loc[i,:])
            track_demand[j].append(demand[i])

        for k in range(loc_center.shape[0]):
            if num_center[k]>0:
                loc_center[k]=loc2_center[k]/num_center[k]
                track_loc[k]=torch.stack(track_loc[k],0)
                track_demand[k]=torch.stack(track_demand[k],0)
            else:
                loc_center[k]=torch.tensor([0.5,0.5])
                track_loc[k]=None
                track_demand[k]=None
    
    if plot:  
        for k in range(K):
            plt.plot(track_loc[k].numpy()[:,0],track_loc[k].numpy()[:,1],'.')
        plt.plot(loc_center[:,0].numpy(),loc_center[:,1].numpy(),'bo')
        plt.show()
        
#     print(num_center,num_center.sum())
    
    return track_loc,track_demand


# 此函数作用是将不同分区的不同长度的点序列合并（在长度较小的后面pad 0）,
#   最后输出为region_num*原来一个分区的shap的最大值。
#暂时没考虑li中某元素为空的情况
#dim 为不同的那一维，stack_dim为需要stack的那一维
#size_n:shape=region_num, 为每一个region原来的长度
def stacks(li,device=None,dim=0,stack_dim=0,min_length=None,constant=0):
    m=max([s.shape[dim] for s in li])
    if min_length is not None:
        m=max(m,min_length)
    li1=[]
    pad_dim=[0]*len(li[0].shape)*2
    size_n=torch.tensor([s.shape[dim] for s in li])
    for s in li:
        pad_dim[len(s.shape)*2-2*dim-1]=m-s.shape[dim]
        li1.append(F.pad(s.to(device),pad_dim,value=constant))
#         print(li1[-1])
#         print(li1[-1].shape)
    return torch.stack(li1,stack_dim),size_n

### 数据集的生成
def generate_vrp_regions(batch_size=1,size=500,K=10,center_depot=True,
                         step=10,beta=0,device=torch.device('cpu')):
    CAPACITIES=50
    
    locss=[]
    demandss=[]
    for i in range(batch_size):
        loc=torch.FloatTensor(size, 2).uniform_(0, 1)
        demand=(torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES
        if center_depot:
            depot= torch.tensor([0.5,0.5])
        else:
            depot= torch.rand(2)
        locs,demands=K_means(K,loc,demand,beta=beta,step=step,plot=False)
        locss=locss+locs
        demandss=demandss+demands
    
    locs,size_n=stacks(locss,device,constant=0.5)
    demands,_ =stacks(demandss,device)
    batch={}
    batch['depot']=depot.expand(locs.shape[0],2).to(device)
    batch['loc']=locs.to(device)
    batch['demand']= demands.to(device)
    return batch,size_n.to(device)
    
    
    
def identity_collate(x):
    return x[0]