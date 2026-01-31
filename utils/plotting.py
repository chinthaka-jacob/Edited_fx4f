import matplotlib.figure

def plot_energy_evolution(timesteps,**energies):
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot(111, title=f"Energy")
    for name,energydata in energies.items():
        ax.plot(timesteps,energydata, label=name)
    ax.legend()
    return fig

def plot_convergence(Re,conv_data):
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot(111, title=f"Re = {Re}")
    ax.loglog(1/conv_data[:,0],conv_data[:,1], label='u_L2')
    ax.loglog(1/conv_data[:,0],conv_data[:,2], label='u_H10')
    ax.loglog(1/conv_data[:,0],conv_data[:,3], label='p_L2')
    ax.loglog(1/conv_data[:,0],conv_data[:,4], label='p_H10')
    ax.legend()
    return fig
