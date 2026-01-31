# The FEniCSx for Flow framework

... is a containerized and compartmentalized [FEniCSx](https://fenicsproject.org/) set-up with [treelog](https://github.com/evalf/treelog) logging system. The [Fx4F](https://gitlab.com/fenicsx4flow/fx4f) repo, together with the [Fx4F Runner](https://gitlab.com/fenicsx4flow/fx4f_runner), forms the basis of the [FEniCSx4Flow](https://gitlab.com/fenicsx4flow/) project. The `Fx4F Runner` repository takes care of the containerization, enabling robust deployment on different computation platforms, through various containerization software (docker, podman, apptainer). The `Fx4F` repository is an importable python module with all supporting code.

For detailed documentation, installation instructions and options, take a look at the [documentation](https://fenicsx4flow.gitlab.io/fx4f/).

## The python module 

The python module collects various CFD related functionality. To facilitate research projects, the intent of this module is to provide general-purpose re-useable code snippets, rather than a full-fletched solver. The module includes, e.g.:
- Reference problems.
- IO (typical plotting routines, snapshot restart).
- Solver and linear solver settings (preconditioning).
- Error computation.
- Aerodynamics metrics/QOI.
- Turbulence statistics.

Detailed documentation of the `Fx4F` functionality can be found [here](https://fenicsx4flow.gitlab.io/fx4f/fx4f.html).

## Quick set-up
Here follow short instructions for getting the [fx4f_bare](https://gitlab.com/fenicsx4flow/fx4f_bare) example simulation running.

1. Clone the [Fx4F Runner](https://gitlab.com/fenicsx4flow/fx4f_runner) repository, and `cd` into it:

``git clone git@gitlab.com:fenicsx4flow/fx4f_runner.git & cd fx4f_runner``

2. **Recursively** clone a particular project into this runner repo:

``git clone --recurse-submodules git@gitlab.com:fenicsx4flow/fx4f_demo.git``

3. (a) Execute the simulation:

``./runner ./fx4f_demo``

3. (b) Or, alternatively, execute a batch run of the simulation

``./runner_batch ./fx4f_demo ./fx4f_demo/input/batch_example``

4. View the html log:

By default, the html logs are now stored inside the `output` directory of the simulation directory: `./fx4f_demo/output/<date+time>/log.html`.
An example log can be viewed [here](https://fenicsx4flow.gitlab.io/fx4f/demo_output/log.html). 

The browser-based log pages can be navigated with certain hotkeys: 
- The `+` and `-` keys fold in/out the log levels,
- When viewing an image the `tab` key toggles between zoom-in and zoom-out to all context level images,
- The `q` key quits viewing images and moves back to the text log,
- The `space` key locks the currectly viewed image, and permits moving back and forth between the same image with the `left` and `right` arrow keys.

## Project structure

A typical directory structure in the FEniCSx4Flow framework looks like:

```
fx4f_runner*               <--- The gitlab.com/fenicsx4flow/fx4f_runner repository
├── docker                 <--- Dockerfiles for different versions
│   └── ...                
├── simulation_project*    <--- Your simulation repository
│   ├── fx4f*              <--- Git-submodule gitlab.com/fenicsx4flow/fx4f
│   │   ├── src/fx4f/      <--- General supporting code (importable as fx4f.*)
│   │   ├── tests/         <--- Unit tests
│   │   ├── pyproject.toml <--- Package configuration
│   │   └── ...            
│   ├── input
│   │   └── ...
│   ├── output
│   │   └── ...
│   ├── utils              <--- Simulation-specific supporting code
│   │   └── ...
│   ├── simulation.py      <--- Your simulation, called by runner script
│   └── postprocessing.py  <--- A batch postprocessing script (optional)  
├── runner
└── runner_batch

* These directories are each a separate git repository.
```

* The **fx4f_runner** repository ([here](https://gitlab.com/fenicsx4flow/fx4f_runner)) provides functionality for containerized running of your simulations on various platforms, and for dispatching batch jobs (parameter sweeps or so).
* The **simulation_project** repository (e.g. [fx4f_bare](https://gitlab.com/fenicsx4flow/fx4f_bare)) is where you develop your code. By default, the `runner` executes either `simulation.py` or `main.py`, although the specific file to be executed can be changed with the `--fname` flag. The `simulation.py` (or `main.py`) file must call a `main` function decorated with the `fx4f.core.fx4fsugar` decorator. See example snippet below.
* The **fx4f** repository ([here](https://gitlab.com/fenicsx4flow/fx4f)) must be included in your simulation repository as a submodule. It includes the utilities for argument parsing, html-logging, and the modular components of FEniCSx for flow computations. In particular, it includes an interface to various reference solutions.

Example minimal `simulation.py` file in the **simulation_project** repository:
```
from treelog4dolfinx import treelog4dolfinx

@treelog4dolfinx
def main():
    pass

if __name__ == "__main__":
    main()
```


## Runner CLI options

From the `fx4f_runner` directory, the two runners can be executed in most bare-bones fashion as:

`./runner <simulation_dir>`

`./runner_batch <simulation_dir> <batch_file>`

### The runner script
The `runner` script permits the following arguments, *after* having specified the simulation directory:

- `-n --np <procs>` : Integer number of processors for mpi execution.
- `-c --container <docker|podman|apptainer>` : Which containization software to use.
- `-C --complex`: Flag for running FEniCSx in complex number mode.
- `-d --outdir <directory>`: The output directory. By default, this is `<simulation_dir>/output`.
- `-D, --description <text>`: Notes added to the top of the log of the simulation. Add quotations when spaces are used.
- `-i --interactive` : Flag for keeping the container open in interactive mode.
- `-o --onscreen` : Flag for enabling `pyvista` rendering to active display. Is always disabled for podman/apptainer containers.
- `   --outputdir <directory>`: The directory passed to the simulation function for storing simulation output. By default, this is `<simulation_dir>/<outdir>/<name>`.
- `-f --fname <file_name>` : The python file to be executed in the container. By default, it looks for `simulation.py` and `main.py`.
- `-N --name <simulation_name>` : The name given to the simulation in terms of the output directory inside `<outdir>` and in the logger. By default, the name is set to a date+timestamp.

Further arguments are forwarded to the python script, and should follow double-dash convention. For example, to forward `Re` as a keyword argument to the `main` function in your `simulation.py` script:

`./runner <simulation_dir> --Re 1000`

### The batch script

The `runner_batch` script requires as its first and second argument the simulation directory and the batch file, respectively. It then spawns multiple instances of the `runner` script. Specification of the arguments supplied to each of these instances is done in the batch file. This file simply comprises the lines of CLI arguments provided to `runner`. For example:

```
# Temporal convergence test
--T 10 --dt 2 --Re 10 --np 4
--T 10 --dt 1 --Re 10 --np 4
--T 10 --dt 0.5 --Re 10 --np 8
#
# Post processing:
wait
--fname postprocess.py
```

The lines starting with hashtags are added as comments to the batch-log output html. The `wait` keyword tells the `runner_batch` script to execute a Linux `wait`. If the earlier simulations were to be run in parallel, this ensures that they all finish before the next line is executed. That final line uses the `--fname` flag of the `runner` script, explained above, to change the target python script.

The `runner_batch` script also collects the output of each run in a separate subdirectory of the batch output directory. I.e., a typical output structure would look like:

```
<outdir>
├── <batch_name>
│   ├── argumentset1
│   │   ├── log
│   │   │   └── html_files
│   │   ├── log.html
│   │   └── Further simulation output
│   ├── argumentset2
│   │   ├── log
│   │   │   └── html_files
│   │   ├── log.html
│   │   └── Further simulation output
│   log.html
└── ...
```

Arguments specified to `runner_batch` are forwarded to `runner` (and hence potentially to your python main function, where they **overwrite** the arguments specified in the batch file).

The exceptions are the following dedicated arguments:

- `-P --parallel` : Flag for whether all argumentsets should be executed simulataneously. By default (=unspecified), they run sequentially.
- `-d --outdir <directory>`: This is now the output directory of the batch as a whole, as illustrated in the structure above. Default: `<simulation_dir>/output`.
- `-D, --description <text>`: Notes added to the top of the log of the simulation. Add quotations when spaces are used.
- `-N --name <batch_name>` : This is now the name given to the batch output directory, as illustrated in the structure above. By default, a date+timestamp is appended to `batch_`.