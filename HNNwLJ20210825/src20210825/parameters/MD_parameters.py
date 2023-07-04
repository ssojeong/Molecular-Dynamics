import json

from integrator.methods import linear_velocity_verlet

class MD_parameters:

    MC_init_config_filename      = None             # filename to read data
    MD_data_dir                  = None             # path to write data
    MD_output_basenames          = None             # filename to write data
    crash_filename               = None             # filename to write crash info
    crash_path                   = None             # path to write data
    crash_checker                = None             # filename to use crash or crash dummy
    crash_thrsh                  = None             # filename to give thrsh (strong or soft crsh)
    
    rthrsh0                      = None
    pthrsh0                      = None
    rthrsh                       = None
    pthrsh                       = None

    tau_short                    = None             # short time step for label
    tau_long                     = None             # value of tau_long

    long_traj_save2file          = None             # save index file
    long_traj_no_crash_save      = None

    append_strike                = None             # number of short steps to make one long step
    save2file_strike             = None             # number of short steps to save to file
    niter_tau_short              = None             # number of MD steps for short tau

    hamiltonian_type             = None

    integrator_method = linear_velocity_verlet.linear_velocity_verlet
    # integrator_method_backward = linear_velocity_verlet.linear_velocity_verlet_backward

    @staticmethod
    def load_dict(json_filename):
        with open(json_filename) as f:
            data = json.load(f)

        MD_parameters.MC_init_config_filename     = data['MC_init_config_filename']
        MD_parameters.MD_data_dir                 = data['MD_data_dir']
        MD_parameters.MD_output_basenames         = data['MD_output_basenames']
        MD_parameters.crash_filename              = data['crash_filename']
        MD_parameters.crash_path                  = data['crash_path']
        MD_parameters.crash_checker               = data['crash_checker']
        MD_parameters.crash_thrsh                 = data['crash_thrsh']        

        MD_parameters.rthrsh0                     = data['rthrsh0']
        MD_parameters.pthrsh0                     = data['pthrsh0']
        MD_parameters.rthrsh                      = data['rthrsh']
        MD_parameters.pthrsh                      = data['pthrsh']

        MD_parameters.tau_short                   = data['tau_short']
        MD_parameters.tau_long                    = data['tau_long']    
        
        MD_parameters.long_traj_save2file         = data['long_traj_save2file']
        MD_parameters.long_traj_no_crash_save     = data['long_traj_no_crash_save']
                    
        MD_parameters.append_strike               = data['append_strike']
        MD_parameters.save2file_strike            = data['save2file_strike']
        MD_parameters.niter_tau_short             = data['niter_tau_short']

        MD_parameters.hamiltonian_type            = data['hamiltonian_type']
