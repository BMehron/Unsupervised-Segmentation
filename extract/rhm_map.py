"""Implementation of optimal transport+geometric post-processing (Hough voting)"""

import math

import torch.nn.functional as F
import torch
from tqdm import tqdm


def apply_sinkhorn(C,eps,mu=None,nu=None, niter=10,tol=10e-9):
    if mu is None:
        mu = torch.ones((C.shape[0],1)) / C.shape[0]
    #mu = mu.cuda()
    if nu is None:
        nu = torch.ones((C.shape[1],1)) / C.shape[1]
    #nu = nu.cuda()
    C = 1 - C
    with torch.no_grad():
        epsilon = eps
        cnt = 0
        while True: # ORIGINAL
        #for _ in tqdm(range(100)): # ADDED BY MEKHRON
            PI,mu,nu,Err = perform_sinkhorn(C, epsilon, mu, nu, niter=niter, tol=tol)
            #PI = sinkhorn_stabilized(mu, nu, cost, reg=epsilon, numItermax=50, method='sinkhorn_stabilized', cuda=True)
            if not torch.isnan(PI).any():
                # if cnt>0:
                    #print(cnt)
                break
            else: # Nan encountered caused by overflow issue is sinkhorn
                epsilon *= 2.0
                #print(epsilon)
                cnt += 1
    PI = torch.clamp(PI, min=0)
    PI = PI * C.shape[0]
    return PI, (Err, mu, nu)

def perform_sinkhorn(C,epsilon,mu,nu,a=[],warm=False,niter=1,tol=10e-9):
    """Main Sinkhorn Algorithm"""
    if not warm:
        a = torch.ones((C.shape[0],1)) / C.shape[0]
        #a = a.cuda()
    K = torch.exp(-C/epsilon).detach().cpu()

    Err = torch.zeros((niter,2))#.cuda()
    for i in range(niter):
        b = nu/torch.mm(K.t(), a)
        if i%2==0:
            Err[i,0] = torch.norm(a*(torch.mm(K, b)) - mu, p=1)
            if i>0 and (Err[i,0]) < tol:
                break

        a = mu / torch.mm(K, b)

        if i%2==0:
            Err[i,1] = torch.norm(b*(torch.mm(K.t(), a)) - nu, p=1)
            if i>0 and (Err[i,1]) < tol:
                break

        PI = torch.mm(torch.mm(torch.diag(a[:,-1]),K), torch.diag(b[:,-1]))

    del a; del b; del K
    return PI,mu,nu,Err



if __name__ == "__main__":
    sim = torch.randn([5,100]).clamp(min=0, max=1)#.cuda() # cosine similarity between [0,1]
    temp = 0.02 # lower temp -> lower entropy -> more sparsity
    n_iter = 10
    out = apply_sinkhorn(sim, temp, niter=40)[0].detach()
    print(sim, out)