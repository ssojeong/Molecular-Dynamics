import math
import torch
from utils.mydevice import mydevice

def thermostat(p_list,gamma,temp,tau):
    # p_list shape [nsamples,nparticles,dim]
    c1 = math.exp(-gamma * 0.5* tau) # weight of the rescaling factor
    c2 = math.sqrt(1 - c1 * c1) * math.sqrt(temp) # weight of the amplitude of the gaussian number
    R = torch.normal(0,1,size=p_list.shape)
    p_new = c1 * p_list + c2 * R
    return p_new

def thermostat_ML(p_list,gamma,temp,tau):
    # p_list shape [nsamples,nparticles,dim]
    c1 = math.exp(-gamma *  tau) # weight of the rescaling factor
    c2 = math.sqrt(1 - c1 * c1) * math.sqrt(temp) # weight of the amplitude of the gaussian number
    R = torch.normal(0,1,size=p_list.shape, device=mydevice.get())
    p_new = c1 * p_list + c2 * R
    return p_new
