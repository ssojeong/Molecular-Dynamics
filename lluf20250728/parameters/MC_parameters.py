import json
import math

class MC_parameters:

    nparticle = None
    temperature = None

    filename  = None                          # filename to write data

    seed = None                               # set different seed for generate data for train/valid/test

    DIM = None
    rho = None                                # density

    nsamples = None                           # the number of samples for mc
    mcstep = None                             # mc step each sample
    max_energy = None                         # energy threshold
    DISCARD = None                            # discard initial mc steps
    interval = None                           # take mc step every given interval
    dq = None                                 # displacement to increase acceptance rate

    boxsize = None
    iterations = None


    @staticmethod
    def load_dict(json_filename):
        with open(json_filename) as f:
            data = json.load(f)

        MC_parameters.nparticle = data['nparticle']
        MC_parameters.temperature = data['temperature']

        MC_parameters.filename  = data['MC_output_filename']
    
        MC_parameters.seed = data['seed']                       # set different seed for generate data for train/valid/test

        MC_parameters.DIM = data['DIM']
        MC_parameters.rho = data['rho']                         # density
    
        MC_parameters.nsamples = data['nsamples']               # the number of samples for mc
        MC_parameters.mcstep = data['mcstep']                   # mc step each sample
        MC_parameters.max_energy = data['max_energy']           # energy threshold
        MC_parameters.DISCARD = data['DISCARD']                 # discard initial mc steps
        MC_parameters.interval = data['interval']               # take mc step every given interval
        MC_parameters.dq = data['dq']                           # displacement to increase acceptance rate

        if MC_parameters.DIM == 2:
            MC_parameters.boxsize = math.sqrt(MC_parameters.nparticle/MC_parameters.rho)
        elif MC_parameters.DIM == 3:
            MC_parameters.boxsize = (MC_parameters.nparticle/MC_parameters.rho)**(1./3)
        else:
            assert False,'dimension of box is limit to 2 or 3  only ..... -.-'
        MC_parameters.iterations = MC_parameters.mcstep*MC_parameters.interval+MC_parameters.DISCARD




    
