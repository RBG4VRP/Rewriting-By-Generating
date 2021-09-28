import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
# import setproctitle
from tqdm import tqdm
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import copy
import random
import time
from datetime import timedelta

from utils import load_model, move_to, load_problem
from utils.data_utils import save_dataset
from utils.functions import parse_softmax_temperature
from nets.attention_model import set_decode_type
from nets.attention_model import AttentionModel
from option import parse


#input: x-
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

def K_means(K,node,beta=0,depot=torch.tensor([0.5,0.5]),step=10,plot=False,device=torch.device('cpu')):
    loc_center=torch.rand(K,2).to(device)
    loc=node[:,:2]

    for n in range(step):
        loc2_center=torch.zeros((K,2)).to(device)
        num_center=torch.zeros(K).to(device)
        track_node=[]  #points for each center, None for zero; shape=[size1*2,size2*2,...,sizeK*2]
#         track_demand=[]  #shape: size1*n
        if n==step-1:
            for i in range(K):
                track_node.append([])
    #             track_demand.append([])

        dis1=torch.argmin(distance1(loc[:,None,:],loc_center,beta,device=device),1) #(N,)
        for i,j in enumerate(dis1):
            loc2_center[j]=loc2_center[j]+loc[i]
            num_center[j]=num_center[j]+1
            if n==step-1:
                track_node[j].append(node[i,:])

        for k in range(K):
            if num_center[k]>0:
                loc_center[k]=loc2_center[k]/num_center[k]
                if n==step-1:
                    track_node[k]=torch.stack(track_node[k],0)   #(num,3)
#                 track_demand[k]=torch.stack(track_demand[k],0)
            else:
                loc_center[k]=torch.rand(2)
                if n==step-1:
                    track_node[k]=None
#                     track_demand[k]=None

    if plot:
        for k in range(K):
            plt.plot(track_node[k].numpy()[:,0],track_node[k].numpy()[:,1],'.')
        plt.plot(loc_center[:,0].numpy(),loc_center[:,1].numpy(),'bo')
        plt.show()

#     print(num_center,num_center.sum())

    return track_node  #[(num_in_cluster, 3)]*K


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


def rotate_matrix(theta,device=torch.device('cpu')):
    cos1=torch.cos(theta) #(batch,)
    sin1=torch.sin(theta)
    Q= torch.stack([torch.stack([cos1,sin1],-1),torch.stack([-sin1,cos1],-1)],1)
    return Q.to(device)

#x:batch*n*2; depot: batch*2
def rotate(x,Q,depot):
    return torch.matmul(Q[:,None,:,:],(x[:,:,:,None]-depot[:,None,:,None]))[:,:,:,0]+depot[:,None,:]


def merge_and_judge_instances(model,regions0,depot,device,cal_ll=False,args=None):
    nodes,size_n=stacks(regions0,device,constant=0) #(K,num_in_cluster,3)
#     demands,_ =stacks(demands,device)
    batch={}
    batch['depot']=depot.expand(nodes.shape[0],2).to(device)
    batch['loc']=nodes[:,:,:2].to(device)
    batch['demand']= nodes[:,:,2].to(device)
    if args.enable_random_rotate_eval:
        theta=torch.rand(nodes.shape[0])*2*3.1415926535
        Q=rotate_matrix(theta,device)
        batch['loc']=rotate(batch['loc'],Q,batch['depot'])
        # print('loc.shape=',batch['loc'].shape)
        # print('loc.shape=',batch['loc'].shape)

    t1=time.time()
    with torch.no_grad():
#         print('size=',size_n.shape,batch['loc'].shape)
        cost1,ll,pi1=model((batch,size_n.to(device)),return_pi=True,cal_ll=cal_ll)
    t2=time.time()
#     print('model_time=',t2-t1)

    return cost1,ll,pi1


def merge_and_train_instances(model,fixed_model,regions0,depot,device,cal_ll=True,args=None):
    model_type=args.model_type
    nodes,size_n=stacks(regions0,device,constant=0) #(K,num_in_cluster,3)
#     demands,_ =stacks(demands,device)
    batch={}
    batch['depot']=depot.expand(nodes.shape[0],2).to(device)
    batch['loc']=nodes[:,:,:2].to(device)
    batch['demand']= nodes[:,:,2].to(device)

    if args.enable_random_rotate_train:
        theta=torch.rand(nodes.shape[0])*2*3.1415926535
        Q=rotate_matrix(theta,device)
        batch['loc']=rotate(batch['loc'],Q,batch['depot'])
        # print('loc.shape=',batch['loc'].shape)

    cost1,ll,pi=model((batch,size_n.to(device)),return_pi=True,cal_ll=cal_ll)



    with torch.no_grad():
        if model_type==0:  #
            cost2,_=model((batch,size_n.to(device)),return_pi=False,cal_ll=False)
        elif model_type==1:
            set_decode_type(model, "greedy")
            cost3,_,_=model((batch,size_n.to(device)),return_pi=True,cal_ll=False)
            set_decode_type(model, "sampling")
            cost2,_,pi=fixed_model((batch,size_n.to(device)),return_pi=True,cal_ll=False)
            # print('cost1=',cost1.mean().item(),'cost2=',cost2.mean().item(),'cost3=',cost3.mean().item())
        elif model_type==2:
            set_decode_type(model, "greedy")
            cost2,_,pi=model((batch,size_n.to(device)),return_pi=True,cal_ll=False)
            set_decode_type(model, "sampling")
        elif model_type==3:
            set_decode_type(fixed_model, "greedy")
            cost2,_=fixed_model((batch,size_n.to(device)),return_pi=False,cal_ll=False)
        else:
            print('undefined model_type:',model_type)
            exit()
    
    return cost1,ll,pi,cost2


#input: regions0-[(num_in_cluster(diff), 3)]*K; pi.shape=(K,100)
#output: [[(_route_len,3)]*route_num_in_a_region]*num_region, 100 points
def pi2regions1(regions0,pi):
    regions1=[]
    assert len(regions0)==pi.shape[0], 'not the same!!'

    pi=pi.cpu()

    for i,r0 in enumerate(regions0):
        pre_j=0
        r1=[]
        for j in pi[i]:
            if pre_j==0:
                pre_j=j
                if j==0:
                    break
                else:
                    r1.append([r0[j-1]])
            else:
                pre_j=j
                if j==0:
                    continue
                r1[-1].append(r0[j-1])
        r1=[torch.stack(route,0) for route in r1]
        regions1.append(r1)
    return regions1


def mean_regions_points(regions1):
    mean=np.zeros((len(regions1),3))
    for i,region in enumerate(regions1):
        c=0
        num=0
        for route in region:
            c+=torch.sum(route,0).numpy()
            num+=route.shape[0]
        mean[i,:]=c/num
    return mean

class SelectModel(nn.Module):
    def __init__(self,embed_dim0=100,lstm_dim=100,embed_dims1=[1000,100],args=None,device=torch.device('cuda')):
        super(SelectModel,self).__init__()
        self.embed_dim0=embed_dim0
        self.lstm_dim=lstm_dim
        self.embed_dims1=embed_dims1
        self.embed_dim=embed_dims1[-1]
        self.linear0=nn.Linear(3,embed_dim0)
        self.LSTMCell=nn.LSTMCell(embed_dim0,lstm_dim)
        self.linears1=[]
        d0=self.lstm_dim
        for d in self.embed_dims1[:-1]:
            self.linears1.append(nn.Linear(d0,d))
            self.linears1.append(nn.ReLU(inplace=True))
            d0=d
        self.linears1.append(nn.Linear(d0,self.embed_dims1[-1]))
        self.linears1=nn.Sequential(*self.linears1)
        self.h0=nn.Parameter(torch.rand(1,lstm_dim, requires_grad=True))
        self.c0=nn.Parameter(torch.rand(1,lstm_dim, requires_grad=True))
        self.near_K=args.near_K
        self.args=args
        self.device=device

    def embed_route(self,route):  #route.shape=(route_cnt,3)
        # print(route.device)
        route_emb=self.linear0(route)
        for i,x in enumerate(route_emb):
            hx,cx=self.LSTMCell(x[None,:],(self.h0,self.c0))
        hx=self.linears1(hx)
        return hx  #shape=(1,emb_dims1[-1])

    def forward(self,regions2):
        region_emb=[]
        for region in regions2:
            hxs=[]
            for route in region:
                hxs.append(self.embed_route(route.to(self.device)))
            hxs=torch.sum(torch.cat(hxs,0),0,keepdim=True)  #(1,emb)
            region_emb.append(hxs)
        region_emb=torch.cat(region_emb,0)  #(regions2_num,emb)
        region_emb=region_emb/torch.norm(region_emb,dim=-1,keepdim=True)   #divide by norm
        cov_matrix=torch.matmul(region_emb,region_emb.transpose(0,1))   #(regions2_num,regions2_num)
        idx=torch.arange(len(regions2))
        cov_matrix[idx,idx]=-np.inf
        # cov_matrix=torch.nn.functional.softmax(cov_matrix,-1)
        # cov_matrix[idx,idx]=0
        # print('cov1=',cov_matrix)
        return cov_matrix  #()

    def merge_regions(self,regions2,cov,costs2,sampling=True):
        near_K=self.near_K
        if near_K is None:
            near_K=len(regions2)-1
        near_K=min(near_K,len(regions2)-1)
        selected=np.zeros(cov.shape[0])
        regions1=[]
        merged_regions2=[]
        costs1=[]
        logits=[]

        mean=mean_regions_points(regions2)  #(num_region,3)
        dist_square=np.sum((mean[:,None,:2]-mean[None,:,:2])**2,-1)
#         print('dist_square.shape=',dist_square.shape)
        arg_near=np.argsort(dist_square,-1)[:,1:(near_K+1)]  #(regions2_num,near_K)
        cov=cov[np.arange(len(regions2))[:,None],arg_near]  #(regions2_num,near_K)
        cov=torch.nn.functional.softmax(cov,-1)
        # cov_matrix[idx,idx]=0
        # print('cov2=',cov)
        if sampling:
            s=torch.multinomial(cov,1)[:,0]    #(regions2_num, )
        else:
            s=torch.argmax(cov,-1)
        s1=s    #(regions2_num, )
        s=arg_near[np.arange(len(regions2)),s.cpu()] #(regions2_num, )

        ranges=list(range(cov.shape[0]))
        random.shuffle(ranges)
        for i in ranges:
            if selected[i] or selected[s[i]]:
                continue
            selected[i]=1
            selected[s[i]]=1
            regions1.append(regions2[i]+regions2[s[i]])
            merged_regions2.append(regions2[i])
            merged_regions2.append(regions2[s[i]])
            costs1.append(costs2[i]+costs2[s[i]])
#             print('cov.shape=',cov.shape,i,s[i])
            logits.append(torch.log(cov[i,s1[i]]))
        regions3=[]
        for i in range(cov.shape[0]):
            if not selected[i]:
                regions3.append(regions2[i])

#         print(logits, costs1)
        return regions1, regions3, selected, torch.stack(logits), torch.tensor(costs1),merged_regions2

def cost_of_route(route,depot):
#     print((route[1:,:2]-route[:-1,:2])**2)
#     print(torch.sum((route[1:,:2]-route[:-1,:2])**2,1))
#     print(torch.sqrt(torch.sum((route[1:,:2]-route[:-1,:2])**2,1)))
#     print(torch.sum(torch.sqrt(torch.sum((route[1:,:2]-route[:-1,:2])**2,1)),0),'\n')
    c=torch.sum(torch.sqrt(torch.sum((route[1:,:2]-route[:-1,:2])**2,1)),0)
    return torch.sqrt(torch.sum((route[0,:2]-depot)**2))+c+torch.sqrt(torch.sum((route[-1,:2]-depot)**2))


def costs_of_regions1(regions1,depot):
    costs=torch.zeros(len(regions1))
    for i,region in enumerate(regions1):
        for route in region:
            costs[i]=costs[i]+cost_of_route(route,depot)
#     print(costs)
    return costs

def plot_regions0(regions0):
    color_base=np.array([0,0,0])
    color_add=np.array([0.15,0.32,0.86])
    def rotate(x):
        x=np.array([a-1 if a>1 else a for a in x])
        return x
    color=color_base
    for region in regions0:
        plt.plot(region[:,0],region[:,1],'.',color=color)
        color=rotate(color+color_add)
    plt.show()


def plot_regions1(regions1,divide_num=1):
    color_base=np.array([0,0,0])
    color_add=np.array([0.15,0.32,0.86])/divide_num
    def rotate(x):
        x=np.array([a-1 if a>1 else a for a in x])
        return x
    color=color_base
    for region in regions1:
        for route in region:
            plt.plot(route[:,0],route[:,1],'.-',color=color)
        color=rotate(color+color_add)
    plt.show()

#
def divide_region(regions1):
    regions2=[]
    for r1 in regions1:  #[(route_len,3)]*route_num
        route_mean=torch.stack([torch.mean(route,0)[:2] for route in r1],0) #(route_num,2)
#         print('route_mean=',route_mean.shape)
        route_cnt=[route.shape[0] for route in r1]   #(route_num,)
        route_total_cnt=sum(route_cnt)
        route_mean_mean=torch.mean(route_mean,0)  #(2,)
        U,S,V=torch.pca_lowrank((route_mean-route_mean_mean))
        route_PCA=torch.matmul((route_mean-route_mean_mean),V[:,:1])[:,0]# (route_num, )
#         print('route_PCA.shape=',route_PCA.shape,route_PCA)
        sorted_route_PCA=torch.argsort(route_PCA)
#         print('sorted_route_PCA=',sorted_route_PCA)
        sub1=[]
        sub2=[]
        cnt=0
        for i,idx in enumerate(sorted_route_PCA):
            if cnt>route_total_cnt/2:
                break
            cnt+=route_cnt[idx]
            sub1.append(r1[idx])

        for idx in sorted_route_PCA[i:]:
            sub2.append(r1[idx])
        regions2.append(sub1)
        regions2.append(sub2)
#     print('len=',len(regions2))
    return regions2

def count_same_num_route(r1,r2):
    cnt=0
    r1=r1.cpu()
    r2=r2.cpu()
    for r11 in r1:
        for r22 in r2:
            if torch.sum(torch.abs(r11-r22))<0.0000001:
                cnt=cnt+1
                break
    return cnt
def count_same_num_region(r1,r2):
#     cnt=0
#     for r11 in r1:
#         for r22 in r2:
#             cnt+=count_same_num_route(r11,r22)
#     return cnt
    cnt=0
    r1=torch.cat(r1,0).cpu()
    r2=torch.cat(r2,0).cpu()
    for r11 in r1:
        for r22 in r2:
            if torch.sum(torch.abs(r11-r22))<0.00001:
                cnt=cnt+1
                break
    return cnt


def count_num_region(region):
    route_len=[r.shape[0] for r in region]
    return sum(route_len)

def divide_and_update_region(regions1,merged_regions2,should_update):
    regions2=[]
    for cnt1,r1 in enumerate(regions1):  #[(route_len,3)]*route_num
        if should_update[cnt1]:
            route_mean=torch.stack([torch.mean(route,0)[:2] for route in r1],0) #(route_num,2)
    #         print('route_mean=',route_mean.shape)
            route_cnt=[route.shape[0] for route in r1]   #(route_num,)
            route_total_cnt=sum(route_cnt)
            route_mean_mean=torch.mean(route_mean,0)  #(2,)
            U,S,V=torch.pca_lowrank((route_mean-route_mean_mean))
            route_PCA=torch.matmul((route_mean-route_mean_mean),V[:,:1])[:,0]# (route_num, )
    #         print('route_PCA.shape=',route_PCA.shape,route_PCA)
            sorted_route_PCA=torch.argsort(route_PCA)
    #         print('sorted_route_PCA=',sorted_route_PCA)
            sub1=[]
            sub2=[]
            cnt=0
            for i,idx in enumerate(sorted_route_PCA):
                if cnt>route_total_cnt/2:
                    break
                cnt+=route_cnt[idx]
                sub1.append(r1[idx])

            for idx in sorted_route_PCA[i:]:
                sub2.append(r1[idx])
        else: #worse than previous solution, therefore not updating
            sub1=merged_regions2[cnt1*2]
            sub2=merged_regions2[cnt1*2+1]

        regions2.append(sub1)
        regions2.append(sub2)
#     print('len=',len(regions2))
    return regions2


def parse_idx(a,idx,dim=0):
    output=[]
    for i in range(len(idx)-1):
        output.append(a[idx[i]:idx[i+1]])
    return output

def generate_data(num_samples=10000,size=500,device=torch.device('cpu')):
    CAPACITIES=50

    return [{'depot':torch.tensor([0.5,0.5]),'loc':torch.FloatTensor(size, 2).uniform_(0, 1),
             'demand':(torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES} for i in range(num_samples)]



def eval(data1,model,selectmodel,args,always_update=False,
         record_iter_steps=None):
    print('evaluation starts.')

    batch=args.eval_batch_size
    iter_step=args.iter_step_eval
    beta=args.beta
    K_means_step=args.K_means_step

    multiple=batch
    select=selectmodel
    K=round(data1[0]['loc'].shape[0]/100)  #for K means
    num_samples=len(data1)

    final_cost=[]
    record_regionss2=[]
    record_costss=[]
    for n in (range(num_samples//multiple+1)):
        t1=time.time()
        if n*multiple==num_samples:
            continue
        loc=[data1[i]['loc'] for i in range(n*multiple,min((n+1)*multiple,num_samples))]   #[(N,2)]*multiple
        depot=[data1[i]['depot'] for i in range(n*multiple,min((n+1)*multiple,num_samples))] # [(N,2)]*multiple
        demand=[data1[i]['demand'] for i in range(n*multiple,min((n+1)*multiple,num_samples))]  #[(N,)]*multiple
        nodes=[torch.cat((loc[i],demand[i][:,None]),-1) for i in range(len(demand))]     #[(N,3)]*multiple

        regionss0=[K_means(K,node,beta=beta,step=K_means_step,plot=False) for node in nodes] #[[(num_in_cluster(diff), 3)]*K]*multiple
#         print('time1=',time.time()-t1)
#         t1=time.time()
#         plot_regions0(regionss0[0])
        idx_regions1=[0]    # [ ]*(multiple+1): start index of each num_regions
        for i in range(len(nodes)):
            idx_regions1.append(idx_regions1[-1]+K)

        regionss2=[[] for i in range(len(nodes))]   #
        remain_regionss2=[[] for i in range(len(nodes))]
        remain_costs=torch.zeros(len(nodes))
        record_costs=[]
        sum_exchange_nums=[]
        for step in range(iter_step):
            regions0=[]
            for regions in regionss0:
                regions0=regions0+regions
#             print('time2=',time.time()-t1)
#             t1=time.time()
            costs1,loss,pi=merge_and_judge_instances(model,regions0,depot[0],device=torch.device('cuda'), cal_ll=False,args=args)
            #costs.shape=(N,); loss.shape=(N,); pi.shape=(N,max_len)
#             print('time3=',time.time()-t1)
#             t1=time.time()
#             print('pi.shape=',pi.shape,pi.device,len(regions0))
            regions1=pi2regions1(regions0,pi)  #[[(_route_len,3)]*route_num_in_a_region]*num_region, 100 points
    #         plot_regions1(regions1)
#             print('time4=',time.time()-t1)
#             t1=time.time()
            if step==0 or always_update:
                regions2=divide_region(regions1)
                costss1=parse_idx(costs1,idx_regions1,dim=0)  #[(num_regions1)]*multiple
            else:
#                 print('costs1.shape=',costs1.shape,merged_costs1.shape,len(regions1))
                should_update=costs1<merged_costs1*(1+args.up_rate_eval) #shape=(num_regions1,)
                assert len(merged_regions2)==len(regions1)*2, 'mismatch of len of regions'
                regions2=divide_and_update_region(regions1,merged_regions2,should_update)
                # sum_exchange_nums.append(sum_exchange_num)
                
                costss1=parse_idx(torch.where(should_update,costs1,merged_costs1),idx_regions1,dim=0)  #[(num_regions1)]*multiple

            costs=torch.tensor([c.sum().item() for c in costss1])+ remain_costs #(multiple,)

            for i in range(len(nodes)):
                regionss2[i]=regions2[idx_regions1[i]*2:idx_regions1[i+1]*2]+remain_regionss2[i]   #[[(_route_len,3)]*route_num_in_a_region]*num_region, 50 points
#             plot_regions1(regions2,divide_num=2)

            merged_costs1=[] #[(region_cnt,)]*multiple
            logits2=[] #[(region_cnt,)]*multiple
            merged_regions2=[]  #[[(route_len,3)]*route_cnt]*(region_num*multiple)
            for i in range(len(nodes)):
                costs2=costs_of_regions1(regionss2[i],depot[i]) #(2K,)
                t1=time.time()
                cov=selectmodel(regionss2[i])
    #             print('len=',len(regionss2[i]))
#                 print('time5=',time.time()-t1)
#                 t1=time.time()
                regions1,regions2, selected, logits2_i, merged_costs1_i, merged_regions2_i=selectmodel.merge_regions(regionss2[i],cov,costs2) #selected.shape=(2K,)
#                 print('time6=',time.time()-t1)
#                 t1=time.time()
                logits2.append(logits2_i)
                merged_costs1.append(merged_costs1_i)
                merged_regions2=merged_regions2+merged_regions2_i
                idx_regions1[i+1]=idx_regions1[i]+len(regions1)
#                 print('costs2=',costs2.sum(),'cost=',costs.sum())
                remain_regionss2[i]=regions2
    #             print('costs2.shape=',costs2.shape,selected.shape)
                remain_costs[i]=torch.sum(costs2)-torch.sum(costs2*selected)

        #         plot_regions1(regions1)
#                 print('len(regions1)=',len(regions1))
        #         plot_regions1(regions2)
#                 print('len(regions2)=',len(regions2))
                regions0=[torch.cat(r,0) for r in regions1]    #[(cnt,3)]*merged_num
                regionss0[i]=regions0
    #         plot_regions0(regions0)

#             t5=time.time()
            logits2=torch.cat(logits2)
            merged_costs1=torch.cat(merged_costs1).to(args.device)
            print('data{},step{}:{}'.format(n,step,costs))#  sample cost,shape=(multiple,)
            if record_iter_steps is not None and step+1 in record_iter_steps:
                record_costs.append(costs)
#             plot_regions1(regionss2[0],divide_num=2)
        final_cost.append(costs)
        record_regionss2=record_regionss2+regionss2
        if record_iter_steps is not None:
            record_costss.append(torch.stack(record_costs,0))
    print('evaluation ends.')
    if record_iter_steps is not None:
        record_costss=torch.cat(record_costss,-1)
        return torch.cat(final_cost), record_regionss2, record_costss
    #     print(regions1)
    else:
        return torch.cat(final_cost), record_regionss2


def clip_grad_norms(param_groups, max_norm=math.inf):

    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_batch(nodes,optimizer,model,fixed_model,selectmodel,fixed_selectmodel,always_update=False,args=None,depot=None,batch_id=None,running_cost=0):
    batch=args.batch_size
    iter_step=args.iter_step_train
    beta=args.beta
    K_means_step=args.K_means_step

    K=round(nodes[0].shape[0]/100)
    # print(K)
    regionss0=[K_means(K,node,beta=beta,step=K_means_step,plot=False) for node in nodes] #[[(num_in_cluster(diff), 3)]*K]*multiple

    total_cost=0  # for multi-step training
    total_logits=0  # for multi-step training
#     plot_regions0(regionss0[0])
    idx_regions1=[0]    # [ ]*(multiple+1): start index of each num_regions
    for i in range(len(nodes)):
        idx_regions1.append(idx_regions1[-1]+K)

    regionss2=[[] for i in range(len(nodes))]
    remain_regionss2=[[] for i in range(len(nodes))]
    remain_costs=torch.zeros(len(nodes))

    for step in range(iter_step):
        regions0=[]
        for regions in regionss0:
            regions0=regions0+regions

        if args.train_selection_only:
            costs1,ll,pi=merge_and_judge_instances(fixed_model,regions0,depot[0],device,cal_ll=True,args=args)
        else:
            costs1_sample,ll,pi,costs1=merge_and_train_instances(model,fixed_model,regions0,depot[0],device,cal_ll=True,args=args)
            #costs.shape=(N,); loss.shape=(N,); pi.shape=(N,max_len)
            loss1=((costs1_sample-costs1)*ll).mean()
            # print('cost_sample=',costs1_sample.mean().item(),'  cost1=',costs1.mean().item())

        regions1=pi2regions1(regions0,pi)  #[[(_route_len,3)]*route_num_in_a_region]*num_region, 100 points
#         plot_regions1(regions1)

        if step==0 or always_update:
            regions2=divide_region(regions1)
            costss1=parse_idx(costs1,idx_regions1,dim=0)  #[(num_regions1)]*multiple

        else:
#             print('costs1.shape=',costs1.shape,merged_costs1.shape,len(regions1))
            # print('!')
            should_update=costs1<merged_costs1*(1+args.up_rate_train)  #shape=(num_regions1,)
            assert len(merged_regions2)==len(regions1)*2, 'mismatch of len of regions'
            regions2=divide_and_update_region(regions1,merged_regions2,should_update)
            
            costss1=parse_idx(torch.where(should_update,costs1,merged_costs1),idx_regions1,dim=0)  #[(num_regions1)]*multiple


        if step>=1:
            if args.enable_running_cost:
                loss2=((costs1-merged_costs1-running_cost)*logits2).mean()
            else:
                loss2=((costs1-merged_costs1)*logits2).mean()
            running_cost=running_cost*args.running_cost_alpha+(1-args.running_cost_alpha)*(costs1-merged_costs1).mean().item()
            # print('running_cost=',running_cost,'costs1=',costs1.mean().item(),', merged_costs1=',merged_costs1.mean().item())
        else:
            loss2=torch.tensor(0)


        optimizer.zero_grad()

        if args.train_selection_only:
            if step>=1:
                loss2.backward()
        else:
            loss=loss1+loss2
            loss.backward()

        with open('logs/log_{}.txt'.format(args.save_model),'a') as f:
            f.write('step={}, loss1={}, loss2={}, running_cost={}\n'.format(step,loss1.item(),loss2.item(),running_cost))

        if args.enable_gradient_clipping:
            grad_norms = clip_grad_norms(optimizer.param_groups, args.max_grad_norm)
            # print(grad_norms)
        optimizer.step()

        costs=torch.tensor([c.sum().item() for c in costss1])+ remain_costs


        for i in range(len(nodes)):
            regionss2[i]=remain_regionss2[i]+regions2[idx_regions1[i]*2:idx_regions1[i+1]*2]   #[[(_route_len,3)]*route_num_in_a_region]*num_region, 50 points
#             plot_regions1(regions2,divide_num=2)


        merged_costs1=[] #[(region_cnt,)]*multiple
        logits2=[] #[(region_cnt,)]*multiple
        merged_regions2=[]  #[[(route_len,3)]*route_cnt]*(region_num*multiple)
        for i in range(len(nodes)):
            costs2=costs_of_regions1(regionss2[i],depot[i]) #(2K,)
            cov=selectmodel(regionss2[i])

#             print('len=',len(regionss2[i]))
            regions1,regions2, selected, logits2_i, merged_costs1_i, merged_regions2_i=selectmodel.merge_regions(regionss2[i],cov,costs2) #selected.shape=(2K,)
            logits2.append(logits2_i)
            merged_costs1.append(merged_costs1_i)
            merged_regions2=merged_regions2+merged_regions2_i
            idx_regions1[i+1]=idx_regions1[i]+len(regions1)
#                 print('costs2=',costs2.sum(),'cost=',costs.sum())
            remain_regionss2[i]=regions2
#             print('costs2.shape=',costs2.shape,selected.shape)
            remain_costs[i]=torch.sum(costs2)-torch.sum(costs2*selected)

    #         plot_regions1(regions1)
#                 print('len(regions1)=',len(regions1))
    #         plot_regions1(regions2)
#                 print('len(regions2)=',len(regions2))
            regions0=[torch.cat(r,0) for r in regions1]    #[(cnt,3)]*merged_num
            regionss0[i]=regions0
#         plot_regions0(regions0)

#             t5=time.time()
        logits2=torch.cat(logits2)
        merged_costs1=torch.cat(merged_costs1).to(device)
        print('data{},step{}:{}'.format(batch_id,step,costs))
    return running_cost
#         plot_regions1(regionss2[0],divide_num=2)
#     final_cost.append(costs)
#     record_regionss2=record_regionss2+regionss2
#     #     print(regions1)
#     return torch.cat(final_cost), record_regionss2
def train(model,fixed_model,selectmodel,fixed_selectmodel,args,val_dataset):

    seed=torch.random.get_rng_state()
    torch.manual_seed(args.seed)
    costs,_=eval(val_dataset,fixed_model,fixed_selectmodel,args,always_update=False)
    torch.random.set_rng_state(seed)
    print('epoch 0, val_cost=',costs.mean())

    best_val_cost=costs.mean()
    best_id=-1
    lr1=args.lr1
    lr2=args.lr2
    running_cost=0
    if args.train_selection_only:
        optimizer=torch.optim.SGD([{'params': selectmodel.parameters(), 'lr': lr2}])
    else:
        optimizer=torch.optim.SGD([{'params': model.parameters(), 'lr': lr1}]+
                         [{'params': selectmodel.parameters(), 'lr': lr2}])

    for epoch in range(num_epochs):
        data=generate_data(num_samples=epoch_size,size=size,device=device)
    #     final_cost=[]
    #     record_regionss2=[]
        if args.train_at_train:
            model.train()
        else:
            model.eval()
        set_decode_type(model, "sampling")
        for n in range(epoch_size//batch_size+1):
            if n*batch_size==epoch_size:
                continue
            loc=[data[i]['loc'] for i in range(n*batch_size,min((n+1)*batch_size,epoch_size))]   #[(N,2)]*multiple
            depot=[data[i]['depot'] for i in range(n*batch_size,min((n+1)*batch_size,epoch_size))] # [(N,2)]*multiple
            demand=[data[i]['demand'] for i in range(n*batch_size,min((n+1)*batch_size,epoch_size))]  #[(N,)]*multiple
            nodes=[torch.cat((loc[i],demand[i][:,None]),-1) for i in range(len(demand))]     #[(N,3)]*multiple

            running_cost=train_batch(nodes,optimizer,model,fixed_model,selectmodel,fixed_selectmodel,args=args,always_update=args.always_update,depot=depot, batch_id=n,running_cost=running_cost)
            # if (n+1)%args.update_model_step==0:
            #     fixed_model.load_state_dict(model.state_dict())
            #     fixed_model.eval()
            #     set_decode_type(fixed_model, "greedy")
        model.eval()
        set_decode_type(model, "greedy")
        # val_dataset=generate_data(num_samples=val_size,size=size)
        # costs,_=eval(val_dataset,model,selectmodel,args,always_update=False )
        seed=torch.random.get_rng_state()
        torch.manual_seed(args.seed)
        costs,_=eval(val_dataset,model,selectmodel,args,always_update=False)
        torch.random.set_rng_state(seed)
        cost1=costs.mean()

        if not args.train_selection_only:
            if args.fix_select_model:
                cost2=best_val_cost
            else:
                seed=torch.random.get_rng_state()
                torch.manual_seed(args.seed)
                costs2,_=eval(val_dataset,fixed_model,selectmodel,args,always_update=False )
                cost2=costs2.mean()
                torch.random.set_rng_state(seed)
            if cost2<cost1:
                pass
            else:
                print('update model!')
                fixed_model.load_state_dict(model.state_dict())
                fixed_model.eval()
                set_decode_type(fixed_model, "greedy")
                fixed_selectmodel.load_state_dict(selectmodel.state_dict())
                best_val_cost=cost1
                best_id=epoch
        print('epoch={}, val_cost={}, val_cost2={}'.format(epoch,cost1.item(),cost2.item()))
        with open('logs/log_{}.txt'.format(args.save_model),'a') as f:
            f.write('epoch={}, val_cost={}, val_cost2={}\n'.format(epoch,cost1.item(),cost2.item()))
        # fixed_model.load_state_dict(model.state_dict())
        # fixed_model.eval()
        # set_decode_type(fixed_model, "greedy")
        if (epoch+1)%args.save_epoch==0:
            with open('saved/{}_{}.pkl'.format(args.save_model, epoch),'wb') as f:
                pickle.dump([model.state_dict(),selectmodel.state_dict()], f)
    with open('saved/{}_{}.pkl'.format(args.save_model, num_epochs),'wb') as f:
        pickle.dump([fixed_model.state_dict(),fixed_selectmodel.state_dict()], f)




if __name__ == '__main__':
    args=parse()

    # import setproctitle
    # setproctitle.setproctitle(args.proctitle)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    with open('logs/log_{}.txt'.format(args.save_model),'w') as f:
        f.write('0\n')

    device=torch.device('cuda:0')
    args.device=device

    path_100=[
        'pretrained/pretrain.pt', 
             ]

    model_100, _ = load_model(path_100[0])
    model_100.to(device)
    model_100.set_decode_type("greedy" ,temp=1)

    epoch_size=args.epoch_size
    num_epochs=args.num_epochs
    batch_size=args.batch_size
    val_size=args.val_size
    size=args.size

    val_dataset=generate_data(num_samples=val_size,size=size,device=device)
    if args.train_from_scratch:
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
        model.set_decode_type("greedy" ,temp=1)
    else:
        model=copy.deepcopy(model_100)
    fixed_model=copy.deepcopy(model)
    fixed_model.eval()
    set_decode_type(fixed_model, "greedy")

    selectmodel=SelectModel(args=args,device=device ).to(device)
    fixed_selectmodel=SelectModel(args=args,device=device ).to(device)
    fixed_selectmodel.load_state_dict(selectmodel.state_dict())
    # costs,_=eval(val_dataset,model,selectmodel,args,always_update=False)
    # print('epoch=0, val_cost=',costs.mean())

    train(model,fixed_model,selectmodel,fixed_selectmodel, args,val_dataset)

