
This example requires ClimateMachine.jl to be installed in the same parent
directory as EnsembleKalmanProcesses.jl. 

You may install ClimateMachine.jl directly from GitHub:

>> git clone https://github.com/CliMA/ClimateMachine.jl.git

The example makes use of a simple single column model configuration with two tunable 
parameters to showcase the use of EnsembleKalmanProcesses.jl with ClimateMachine.jl.

This example is a perfect model experiment, in the sense that the ground truth is 
generated using the same model and a certain combination of parameters. The parameters
used to generate the ground truth are (C_smag, C_drag) = (0.21, 0.0011). The simulation
results are strongly influenced by C_smag, and very weakly by C_drag. Thus, we expect
the EKP to recover C_smag. 

To run the example using a SLURM workload manager, simply do:

>> sbatch ekp_calibration.sbatch

This will create a queue on the workload manager running,

  1. ekp_init_calibration (creating initial parameters),
  2a. ekp_single_cm_run (running the forward model with the given parameters),
  2b. ekp_cont_calibration (updating parameters with EKP given the model output).

Steps 2a and 2b are iterated until the desired number of iterations
(defined in ekp_calibration.sbatch) is reached.

The results for different runs of ClimateMachine.jl will be stored in NetCDF format in directories
identifiable by their version number. Refer to the files version_XX.txt to identify each run with
the XX iteration of the Ensemble Kalman Process. To aggregate the parameter ensembles generated during
the calibration process, you may use the agg_clima_ekp(...) function located in helper_funcs.jl.

ClimateMachine.jl is a rapidly evolving software and this example may stop working in the future. If you
find this to be the case, please contact Ignacio Lopez Gomez at ilopezgo@caltech.edu.
