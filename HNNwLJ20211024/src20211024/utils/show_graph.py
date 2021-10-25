import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

class show_graph:

    @staticmethod
    def u_fluctuation(e, temp, nparticle):
        '''plot of potential energy at different temp or combined temp for nsamples ( train/ valid ) '''

        plt.title('mc steps at T={}'.format(temp),fontsize=15)
        plt.plot(e,'k-', label = 'potential energy on {} particles'.format(nparticle))
        plt.xlabel('mc step',fontsize=20)
        plt.ylabel(r'$U_{ij}$',fontsize=20)
        plt.legend()
        plt.show()

    @staticmethod
    def compare2energy(e1, e2, temp, nsample1, nsample2):

        plt.title('mc steps appended to nsamples given at T={}'.format(temp),fontsize=15)
        plt.plot(e1,'k-', label = 'potential energy using {} samples for training'.format(nsample1))
        plt.plot(e2,'r-', label = 'potential energy using {} samples for training'.format(nsample2))
        plt.xlabel('mcs',fontsize=20)
        plt.ylabel(r'$U_{ij}$',fontsize=20)
        plt.legend()
        plt.show()

    @staticmethod
    def u_distribution4nsamples(u, rho, nparticle, boxsize, nsample):
        '''plot of energy distribution at different temp or combined temp for nsamples 

        parameter
        -----------
        u           : potential energy for data
        nsamples    : the number of data
        '''

        # plt.xlim(xmin=-5.1, xmax = -4.3)
        fig, ax = plt.subplots()
        plt.hist(u.numpy(), bins=100, color='k', alpha = 0.5, label='n={} rho={}'.format(nparticle, rho))
        plt.xlabel(r'$U_{ij}$',fontsize=30)
        plt.ylabel('hist', fontsize=30)
        # anchored_text = AnchoredText('npar={} boxsize={:.3f} rho={} no.of data = {}'.format(nparticle, boxsize, rho, nsample), loc= 'upper left', prop=dict(fontweight="normal", size=12))
        # ax.add_artist(anchored_text)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.legend(fontsize=30)
        plt.show()

