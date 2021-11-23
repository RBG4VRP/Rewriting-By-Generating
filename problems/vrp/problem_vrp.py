
from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np

from problems.vrp.state_cvrp import StateCVRP
from problems.vrp.state_sdvrp import StateSDVRP
from utils.beam_search import beam_search
from utils import generate_vrp_regions


class CVRP(object):

    NAME = 'cvrp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        
        # sorted_pi = pi.data.sort(1)[0]
        mask=(pi==0)
        mask[:,:-1]=mask[:,:-1] & mask[:,1:]

        # Sorting it should give all zeros at front and then 1...n
        #assert (
        #    torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
        #    sorted_pi[:, -graph_size:]
        #).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -CVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        d = demand_with_depot.gather(1, pi)
        

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for i in range(pi.size(1)):
            used_cap += d[:, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
            # Cannot use less than 0
            used_cap[used_cap < 0] = 0
            assert (used_cap <= CVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), mask

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = CVRP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class SDVRP(object):

    NAME = 'sdvrp'  # Split Delivery Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi):
        batch_size, graph_size = dataset['demand'].size()

        # Each node can be visited multiple times, but we always deliver as much demand as possible
        # We check that at the end all demand has been satisfied
        demands = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -SDVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )
        rng = torch.arange(batch_size, out=demands.data.new().long())
        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        a_prev = None
        for a in pi.transpose(0, 1):
            assert a_prev is None or (demands[((a_prev == 0) & (a == 0)), :] == 0).all(), \
                "Cannot visit depot twice if any nonzero demand"
            d = torch.min(demands[rng, a], SDVRP.VEHICLE_CAPACITY - used_cap)
            demands[rng, a] -= d
            used_cap += d
            used_cap[a == 0] = 0
            a_prev = a
        assert (demands == 0).all(), "All demand must be satisfied"

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return (
            (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
            + (d[:, 0] - dataset['depot']).norm(p=2, dim=1)  # Depot to first
            + (d[:, -1] - dataset['depot']).norm(p=2, dim=1)  # Last to depot, will be 0 if depot is last
        ), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateSDVRP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"
        assert not compress_mask, "SDVRP does not support compression of the mask"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = SDVRP.make_state(input)

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    }


class VRPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(VRPDataset, self).__init__()
        # if distribution[0:7]=='regions':
        #     self.region_data=True

        # self.data_set = []
        # if filename is not None:
        #     print('load dataset!!!')
        #     assert os.path.splitext(filename)[1] == '.pkl'

        #     with open(filename, 'rb') as f:
        #         data = pickle.load(f)
        #         l1=len(data)
        #     if self.region_data:
        #         #offset is the num of epoches
        #         t1=l1//num_samples
        #         offset=offset%t1
        #         offset=offset*num_samples
        #         self.data = data[offset:offset+num_samples]
        #         str1=distribution.split('_')
        #         self.batchsize=int(str1[1])
        #         self.size1=int(str1[2])
        #         self.K=int(str1[3])
        #         self.step=int(str1[4])
        #         self.beta=float(str1[5])
        #     else:
        # #         self.data = [make_instance(args) for args in data[offset:offset+num_samples]]

        # else:
            
        CAPACITIES = {
            10: 20.,
            20: 30.,
            50: 50.,
            100: 50.,
            500:50.,
            1000:50.,
            2000:50.
        }

        # if distribution==None:
        #     self.data = [
        #         {
        #             'loc': torch.FloatTensor(size, 2).uniform_(0, 1),
        #             # Uniform 1 - 9, scaled by capacities
        #             'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
        #             'depot': torch.FloatTensor(2).uniform_(0, 1)
        #         }
        #         for i in range(num_samples)
        #     ]
        ## distribution=regions_batchsize_size_K_step_beta_device

        # elif distribution[0:7]=='regions':
        #     print('region data')
        #     str1=distribution.split('_')
        #     self.batchsize=int(str1[1])
        #     self.size1=int(str1[2])
        #     self.K=int(str1[3])
        #     self.step=int(str1[4])
        #     self.beta=float(str1[5])
        #     self.device=torch.device(str1[6])
        #     self.size=num_samples
        #     print('generating data:{}'.format(num_samples))
        # #     self.data=[ generate_vrp_regions(batch_size=self.batchsize,size=self.size1,K=self.K,center_depot=True,step=self.step,beta=self.beta,device=self.device) for i in range(num_samples) ]
        # #     print('done!')

        # else:
            # setting=distribution.split('_')
            # if setting[0]=='1':
            #     c1=torch.rand(num_samples,2,dtype=torch.float)*0.8+0.1
            #     sigma1=(torch.min(torch.stack([c1,1-c1]),0))[0]/2
            #     # print(len(sigma1))
            #     #sigma1=torch.from_numpy(sigma1,dtype=torch.float)
            #     #self.data=[generate_vrp_regions(batch_size=self.batchsize,size=self.size1,K=self.K,center_depot=True,step=self.step,beta=self.beta,device=self.device) for i in range(num_samples)]

            # elif setting[0]=='2':
        k1=0 #10 for uniform
        c1=torch.rand(num_samples,2,dtype=torch.float)*0.8+0.1
        c2=torch.rand(num_samples,2,dtype=torch.float)*0.8+0.1
        sigma1=(torch.min(torch.stack([c1,1-c1]),0))[0]/2
        sigma2=(torch.min(torch.stack([c2,1-c2]),0))[0]/2
        # print(len(sigma1))
        #sigma1=torch.from_numpy(sigma1,dtype=torch.float)
        self.data=[
            {
            'loc': torch.cat([torch.randn((size-k1)//2,2,dtype=torch.float)*sigma1[i,:]+c1[i,:],torch.randn((size-k1)//2,2,dtype=torch.float)*sigma2[i,:]+c2[i,:],torch.rand(k1,2,dtype=torch.float)],0),
            # Uniform 1 - 9, scaled by capacities
            'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
            'depot': torch.FloatTensor(2).uniform_(0, 1)
            }
            for i in range(num_samples)
        ]

        self.size = len(self.data)

    def __len__(self):
        #print('len=',self.size)
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
