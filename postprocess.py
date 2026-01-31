import numpy as np
from treelog4dolfinx import *
from pathlib import Path
import pickle
from utils.plotting import plot_convergence
import os

@treelog4dolfinx # Wrap in logging environment and handle argument parsing
def main(outputdir:str = "", # Directory for retrieving output
    ):
    """
    Plotting convergence curves from exported error values
    """
    batchdir = Path(outputdir).resolve().parents[0]
    rundirs = [d for d in os.listdir(batchdir) if os.path.isdir(os.path.join(batchdir, d)) and not d==Path(outputdir).name]

    # Read solution data
    data = []
    for rundir in rundirs:
        with open(batchdir / rundir / 'postprocessdata.pickle', 'rb') as handle:
            data.append(pickle.load(handle))

    # Parse data
    Re = data[0]["Re"]
    label_order = ["u_L2", "u_H10", "p_L2", "p_H10"]
    conv_data = np.zeros((len(data),5))
    for i,sim_output in enumerate(data):
        conv_data[i,0] = sim_output['dt']
        for j,label in enumerate(label_order):
            conv_data[i,j+1] = sim_output[label]
    conv_data = conv_data[(-conv_data[:, 0]).argsort()] # Sort based on timestep size

    # Convenient logging of error values for copy-pasting
    log(f'dt = {conv_data[:,0].tolist()}')
    for label_index,label in enumerate(label_order):
        log(f'{label} = {conv_data[:,label_index+1].tolist()}')

    # Log convergence orders
    for label_index,label in enumerate(label_order):
        with logcontext(f"Convergence orders {label}"):
            for i in range(len(data)-1):
                dt_factor = conv_data[i,0]/conv_data[i+1,0]
                e_factor = conv_data[i,label_index+1]/conv_data[i+1,label_index+1]
                log(np.emath.logn(dt_factor,e_factor) )

    # Log the convergence plot
    log_figure_mpl(f"Convergence.png",plot_convergence(Re, conv_data))


if __name__ == "__main__":
    main()