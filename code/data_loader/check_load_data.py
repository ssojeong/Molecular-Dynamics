import torch
from utils.pbc import pbc
from hamiltonian.lennard_jones2d import lennard_jones2d
#from MD.velocity_verlet_MD import velocity_verlet_MD
import time

class check_load_data:

    def __init__(self,qpl_list_init, qpl_list_final):

        self.q_list_init = qpl_list_init[:,0,:,:,:]
        # q_list_init.shape = [nsamples, trajectory, nparticles, DIM]
        self.p_list_init = qpl_list_init[:,1,:,:,:]
        self.q_list_final = qpl_list_final[:,0,:,:]
        # q_list_final.shape = [nsamples, nparticles, DIM]
        self.p_list_final = qpl_list_final[:,1,:,:]
        self.l_list = qpl_list_init[:,2,0,:,:]
        # l_list.shape is [nsamples,nparticles,DIM]

        self.nsamples, self.trajectory, self.nparticles, _ = self.q_list_init.shape
        self.potential_function = lennard_jones2d()
        #self.mdvv = velocity_verlet_MD(self.potential_function)

    # ==========================================================
    def check(self,tau_short):
        self.check_distance()
        #self.check_force(tau_short)
        self.max_lj_energy()
        self.delta_total_energy()
        #self.delta_momentum()
        self.check_boxsize(self.q_list_init, self.p_list_init,self.l_list)

    # ==========================================================
    def md_trajectory(self,q_init,p_init,q_final,p_final,l_list,label_idx,tau,nitr,append_strike):

        _, nsamples, nparticles, DIM = q_init.shape
        q_init = q_init.clone().detach()
        p_init = p_init.clone().detach()
        l_list = l_list.clone().detach()

        for t in range(1,label_idx+1):
            start_time = time.time()
            print('data_loader/check_load_data.py: label_idx ',label_idx,' t ',t,flush=True)
            qpl_list = self.mdvv.nsteps(q_init[t-1],p_init[t-1],l_list,tau,nitr,append_strike)
            qpl_list = torch.cat(qpl_list, dim=0) # shape [nsamples,3,nparticles,dim]
            q_nxt = qpl_list[:,0,:,:] # shape [nsamples,nparticles,dim]
            p_nxt = qpl_list[:,1,:,:]

            if t < label_idx:
                dq = pbc((q_nxt - q_init[t]),l_list)
                dp = p_nxt - p_init[t]
            else:
                dq = pbc((q_nxt - q_final),l_list)
                dp = p_nxt - p_final

            dq2 = torch.mean(dq*dq).item()
            dp2 = torch.mean(dp*dp).item()
            d2 = dq2+dp2

            print("nitr {}, --- {:.3f} seconds , diff: {:3e} ---".format(nitr, time.time() - start_time, d2))
            assert (dq2 < 1e-8 and dp2 < 1e-8), print('error ....  difference btw states too big', 'dq', dq2, 'dp', dp2)
        print('difference between md trajectory and prepared trajectory is correct .... ')
        return True # for external assert

    def max_lj_energy(self):
        pe = self.potential_function.total_energy(self.q_list_final, self.l_list)
        print('maximum pe', torch.max(pe), 'element', torch.where(pe == torch.max(pe)))

    def check_distance(self):
        print('check min pairwise distance ...')
        dr = self.potential_function.paired_distance(self.q_list_final, self.l_list)
        print('min distance {:.3f}'.format(torch.min(dr)))
        #print('save pairwise dr... filename:  dr_s{}.pt'.format(dr.shape[0]))
        #torch.save(dr, 'dr_s{}.pt'.format(dr.shape[0]))

    def check_force(self,tau):
        print('check force at initial... tau',tau)
        self.mdvv.one_step(self.q_list_init[:,0,:,:],self.p_list_init[:,0,:,:],self.l_list,tau)
        print('check force at final... tau', tau)
        self.mdvv.one_step(self.q_list_final, self.p_list_final,self.l_list,tau)

    def delta_total_energy(self):

        e_init = []
        for traj in range(self.trajectory):
            pe_init = self.potential_function.total_energy(self.q_list_init[:,traj,:,:], self.l_list)
            ke_init = torch.sum(self.p_list_init[:,traj,:,:] * self.p_list_init[:,traj,:,:], dim=(1, 2)) * 0.5
            e_init.append(pe_init + ke_init)
        e_init = torch.stack(e_init,dim=1)

        pe_final = self.potential_function.total_energy(self.q_list_final, self.l_list)
        ke_final = torch.sum(self.p_list_final * self.p_list_final, dim=(1, 2)) * 0.5

        e_all = torch.cat((e_init,torch.unsqueeze(pe_final + ke_final,dim=1)),dim=1)
        e_all0 = torch.unsqueeze(e_all, dim=1)
        estatm = torch.repeat_interleave(e_all0, e_all.shape[1], dim=1)
        etatet = estatm.permute(0, 2, 1)
        de = etatet - estatm
        de_pair = torch.max(de*de)

        print("maximum difference btw initial and final energy not big ...",de_pair)
        assert( de_pair < 1e-3) , "error ... difference btw initial and final energy too big"

    def delta_momentum(self):

        p_init = torch.sum(self.p_list_init, dim=2) # [nsamples, trajectory, DIM]
        p_final = torch.sum(self.p_list_final, dim=1) # [nsamples, DIM]
        p_all = torch.cat((p_init, torch.unsqueeze(p_final,dim=1)),dim=1) # [nsamples, trajectory, DIM]
        px_all0 = torch.unsqueeze(p_all, dim=1)
        pxstatm = torch.repeat_interleave(px_all0, p_all.shape[1], dim=1) # [nsamples, trajectory, trajectory, DIM]
        pxtatet = pxstatm.permute(0, 2, 1, 3)
        dpx = pxtatet - pxstatm
        dpx_pair = torch.max(dpx*dpx)

        assert( dpx_pair < 1e-4) , "error ... difference btw initial and final energy too big"
        print("difference btw initial and final momentum not big ...",dpx_pair)

    def check_boxsize(self, q_init,p_init,l_init):

        # check that l_init is of square box
        lx = l_init[:,:,0]
        ly = l_init[:,:,1]
        lxly = torch.eq(lx,ly)
        assert torch.any(lxly),'some boxes not square'

        # check that q is within box
        for traj in range(self.trajectory):
            assert torch.any(torch.abs(q_init[:,traj,:,:])<0.5*l_init),'particle out of box'

        p_max = 1e3
        for traj in range(self.trajectory):
            assert torch.any(torch.abs(p_init[:,traj,:,:])<p_max),'momentum out of range'

