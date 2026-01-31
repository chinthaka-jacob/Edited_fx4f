# John2004 Reference Data

This directory contains reference data extracted from figures in John (2004)

## File Naming Convention

Data files should be named: `fig{N}_{METHOD}.txt`

Where:
- `{N}` is the figure number (1 or 2)
- `{METHOD}` is the time-stepping method name

## Available Figures

### Figure 1: Re = 1
Error convergence plots for various time-stepping methods at Reynolds number 1.

### Figure 2: Re = 1000  
Error convergence plots for various time-stepping methods at Reynolds number 1000.

## Time-Stepping Methods

The paper compares the following methods:
- **BWE**: Backward Euler
- **CN**: Crank-Nicolson
- **FS0**: Fractional-step scheme (order 0)
- **FS1**: Fractional-step scheme (order 1)
- **ROS3P**: Rosenbrock method (3rd order, P variant)
- **ROWDAIND2**: ROWDA index-2 method
- **ROS3Pw**: Rosenbrock 3P with w-transformation
- **ROS34PW2**: Rosenbrock method (orders 3/4, variant 2)
- **ROS34PW3**: Rosenbrock method (orders 3/4, variant 3)

## File Format

Each data file should contain numerical data in plain text format readable by `numpy.loadtxt()`.

Typical format (space or tab separated):
```
# timestep  velocity_error  pressure_error
0           value           value
1           value           value
2           value           value
...
```

## Usage Example

```python
from fx4f.reference_solutions.John2004 import John2004_1

# Initialize reference solution
refsol = John2004_1(nu=1.0)

# Get all methods for Figure 1 (Re=1)
fig1_data = refsol.get_reference_data("figure_data", fig=1)

# Get specific method
cn_data = refsol.get_reference_data("figure_data", fig=1, method="CN")
print(f"Available methods: {list(fig1_data.keys())}")
```

## Data Extraction

To extract data from the paper figures, the WebPlotDigitizer can be used.