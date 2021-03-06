#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch

class LJ_term:
    def __init__(self, epsilon, sigma, boxsize):

        try:
            self._epsilon  = float(epsilon)
            self._sigma    = float(sigma)
            self._boxsize  = float(boxsize)

        except :
            raise Exception('sigma / epsilon / boxsize error')

        self._name = 'Lennard Jones Potential'

    def phi_npixels(self, xi_space, grid_state):

        xi_state = xi_space.get_q()

        term = torch.zeros((xi_state.shape[0], grid_state.shape[0])) # nsamples x npixels
        nsamples, nparticle, DIM = xi_state.shape
        npixels, DIM = grid_state.shape

        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / pow(self._boxsize, 12)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / pow(self._boxsize, 6)

        for z in range(nsamples):
            for j in range(npixels):

                pair_wise = torch.cat((grid_state[j].unsqueeze(0), xi_state[z]), 0)

                _, d = xi_space.paired_distance_reduced(pair_wise, nparticle+1, DIM) # concat one pixel so that nparticle+1
                d_grid = d[0] # pair-wise between gird and particles

                s12 = 1 / pow(d_grid, 12)
                s6 = 1 / pow(d_grid, 6)

                term[z][j] = torch.sum(a12 * s12 - a6 * s6)

        return term

    def energy(self, xi_space):

        xi_state = xi_space.get_q()
        term = torch.zeros(xi_state.shape[0])

        nsamples, nparticle, DIM  = xi_state.shape

        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / pow(self._boxsize, 12)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / pow(self._boxsize, 6)

        for z in range(nsamples):

            _, d = xi_space.paired_distance_reduced(xi_state[z], nparticle, DIM)

            s12 = 1 / pow(d,12)
            s6  = 1 / pow(d,6)

            term[z] = torch.sum(a12* s12 - a6* s6) * 0.5

        return term

    def evaluate_derivative_q(self,xi_space):

        xi_state = xi_space.get_q()
        dphidxi = torch.zeros(xi_state.shape) # derivative terms of nsamples
        nsamples, nparticle, DIM  = xi_state.shape

        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / pow(self._boxsize, 13)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / pow(self._boxsize, 7)

        for z in range(nsamples):

            delta_xi, d = xi_space.paired_distance_reduced(xi_state[z],nparticle,DIM)
            d = torch.unsqueeze(d,dim =2)
            # print('d',d)
            # print('d*boxsize', d*self._boxsize)
            s12 = -12 * (delta_xi) / pow(d,14)
            s6  = -6 * (delta_xi) / pow(d,8)

            dphidxi[z] = a12*torch.sum(s12, dim=1) - a6*torch.sum(s6, dim=1) # np.sum axis=1 j != k

        # print('dphidxi', dphidxi)

        return dphidxi

    def evaluate_second_derivative_q(self,xi_space):

        xi_state = xi_space.get_q()
        d2phidxi2_append = []

        nsamples, nparticle, DIM  = xi_state.shape
        d2phidxi2 = torch.zeros((nsamples, nparticle * DIM, nparticle * DIM)) # second derivative terms of nsamples
        d2phidxi_lk = torch.zeros((2,2))

        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / pow(self._boxsize, 14)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / pow(self._boxsize, 8)

        for z in range(nsamples):

            d2phidxi2_ = torch.empty((0, nparticle * DIM))

            delta_xi, d = xi_space.paired_distance_reduced(xi_state[z],nparticle,DIM)
            d = torch.unsqueeze(d,dim=2)

            s12_same_term = 1. / pow(d,14)
            s12_lxkx_lyky = (-14) * torch.pow(delta_xi,2) / torch.pow(d,2)
            s12_lxky_lykx = 1. / pow(d,16)

            s6_same_term = 1. / pow(d,8)
            s6_lxkx_lyky = (-8) * torch.pow(delta_xi,2) / torch.pow(d,2)
            s6_lxky_lykx = 1. / pow(d,10)

            for l in range(nparticle):
                j = 0
                for k in range(nparticle):

                    if l == k:
                        d2phidxi_lxkx = a12 *(-12)* (torch.sum(s12_same_term[k] * torch.unsqueeze(s12_lxkx_lyky[k,:,0],dim=-1) + s12_same_term[k],dim=0)) \
                                        - a6 *(-6)* (torch.sum(s6_same_term[k] * torch.unsqueeze(s6_lxkx_lyky[k,:,0],dim=-1) + s6_same_term[k],dim=0))

                        d2phidxi_lxky = a12 * (-12)*(-14)*(torch.sum(s12_lxky_lykx[k]*torch.unsqueeze(delta_xi[k,:,0],dim=-1)*torch.unsqueeze(delta_xi[k,:,1],dim=-1),dim=0))  \
                                        - a6 * (-6)*(-8)*(torch.sum(s6_lxky_lykx[k]*torch.unsqueeze(delta_xi[k,:,0],dim=-1)*torch.unsqueeze(delta_xi[k,:,1],dim=-1),dim=0))

                        d2phidxi_lykx = a12 * (-12)*(-14)*(torch.sum(s12_lxky_lykx[k]*torch.unsqueeze(delta_xi[k,:,0],dim=-1)*torch.unsqueeze(delta_xi[k,:,1],dim=-1),dim=0)) \
                                        - a6 * (-6)*(-8)*(torch.sum(s6_lxky_lykx[k]*torch.unsqueeze(delta_xi[k,:,0],dim=-1)*torch.unsqueeze(delta_xi[k,:,1],dim=-1),dim=0))

                        d2phidxi_lyky = a12 *(-12)* (torch.sum(s12_same_term[k] * torch.unsqueeze(s12_lxkx_lyky[k,:,1],dim=-1) + s12_same_term[k],dim=0)) \
                                        - a6 *(-6)* (torch.sum(s6_same_term[k] * torch.unsqueeze(s6_lxkx_lyky[k,:,1],dim=-1) + s6_same_term[k],dim=0))

                        d2phidxi_lk = torch.tensor((d2phidxi_lxkx[0], d2phidxi_lxky[0], d2phidxi_lykx[0], d2phidxi_lyky[0])).reshape(2, 2)
                        print('l=k d2phidxi_lk',d2phidxi_lk)

                    if l != k:
                        print('j',j)

                        d2phidxi_lxkx = - a12 *(-12)* s12_same_term[l][j] *( s12_lxkx_lyky[l][j][0] + 1) \
                                        + a6 *(-6)* s6_same_term[l][j] * ( s6_lxkx_lyky[l][j][0] + 1)

                        d2phidxi_lxky = - a12 * (-12)*(-14) * (s12_lxky_lykx[l][j] * delta_xi[l][j][0] * delta_xi[l][j][1]) \
                                        + a6 * (-6)*(-8) * (s6_lxky_lykx[l][j] * delta_xi[l][j][0] * delta_xi[l][j][1])

                        d2phidxi_lykx = - a12 * (-12)*(-14)*(s12_lxky_lykx[l][j] * delta_xi[l][j][0]* delta_xi[l][j][1]) \
                                        + a6 * (-6)*(-8)* (s6_lxky_lykx[l][j] * delta_xi[l][j][0] * delta_xi[l][j][1])

                        d2phidxi_lyky = - a12 *(-12)* s12_same_term[l][j] * ( s12_lxkx_lyky[l][j][1]  + 1) \
                                        + a6 *(-6)* s6_same_term[l][j]  * ( s6_lxkx_lyky[l][j][1] + 1)

                        d2phidxi_lk = torch.tensor((d2phidxi_lxkx[0],d2phidxi_lxky[0],d2phidxi_lykx[0],d2phidxi_lyky[0])).reshape(2,2)
                        print('l != k d2phidxi_lk',d2phidxi_lk)

                        j = j + 1
                    d2phidxi2_append.append(d2phidxi_lk)

                if k == nparticle - 1:
                    temp = torch.stack(d2phidxi2_append,dim=0)
                    temp = temp.permute((1,0,2)).reshape(2, nparticle * DIM )
                    print('reshape',temp)
                    d2phidxi2_append = []

                d2phidxi2_ = torch.cat((d2phidxi2_,temp),dim=0)

            d2phidxi2[z] = d2phidxi2_
        print('d2phidxi2', d2phidxi2)

        return d2phidxi2
