import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

class show_graph:

    @staticmethod
    def u_fluctuation(e, temp, nsample, mode):
        '''plot of potential energy at different temp or combined temp for nsamples ( train/ valid ) '''

        plt.title('mc steps appended to nsamples given at T={}'.format(temp),fontsize=15)
        plt.plot(e,'k-', label = 'potential energy using {} samples for {}'.format(nsample, mode))
        plt.xlabel('mcs',fontsize=20)
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
    def u_distribution4nsamples(u, temp, nparticle, boxsize, nsample):
        '''plot of energy distribution at different temp or combined temp for nsamples 

        parameter
        -----------
        u           : potential energy for data
        nsamples    : the number of data
        '''

        # plt.xlim(xmin=-5.1, xmax = -4.3)
        fig, ax = plt.subplots()
        plt.hist(u.numpy(), bins=100, color='k', alpha = 0.5, label = 'histogram of {} samples'.format(nsample))
        plt.xlabel(r'$U_{ij}$',fontsize=20)
        plt.ylabel('hist', fontsize=20)
        anchored_text = AnchoredText('npar={} boxsize={:.3f} temp={} no.of data = {}'.format(nparticle, boxsize, temp, nsample), loc= 'upper left', prop=dict(fontweight="normal", size=12))
        ax.add_artist(anchored_text)

        plt.legend()
        plt.show()

