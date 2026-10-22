# CryoEvap-v2
A Python software package for simulating cryogenic liquid evaporation in storage tanks under dynamic ambient temperature conditions.

## Installation

### Option 1: Using Conda (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/fdt-uc-chile/CryoEvap-Tair.git
   cd CryoEvap-Tair
   ```

2. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate CryoEvap
   ```

### Option 2: Using pip

1. Clone the repository:
   ```bash
   git clone https://github.com/fdt-uc-chile/CryoEvap-Tair.git
   cd CryoEvap-Tair
   ```

2. Install the package:
   ```bash
   pip install .
   ```

## Requirements

- Python >= 3.12
- NumPy >= 2.2.3
- Pandas >= 2.2.3
- Matplotlib >= 3.10.1
- SciPy >= 1.15.2
- CoolProp == 6.6.0
- IPython (for notebooks)
- psutil (for notebooks)

## Quick Start

```python
from cryoevap.storage_tanks import Tank
from cryoevap.cryogens import Cryogen

# Step 1: Initialize tank geometry and initial conditions
tank = Tank(
    d_i=0.201,      # Inner diameter (m)
    d_o=0.204,      # Outer diameter (m) 
    V_tank=6.75e-3, # Tank volume (m³)
    LF=0.9          # Initial liquid filling fraction (90%)
)

# Step 2: Initialize cryogenic fluid properties
nitrogen = Cryogen(name="nitrogen")
nitrogen.set_coolprops(100000)  # Operating pressure (Pa)
tank.cryogen = nitrogen

# Step 3: Configure heat transfer properties
tank.set_HeatTransProps(
    U_L=0.008,        # Liquid overall heat transfer coefficient (W/m²K)
    U_V=0.008,        # Vapor overall heat transfer coefficient (W/m²K)
    T_air=278.15,     # Ambient air temperature (K)
    Q_b_fixed=None,   # Bottom heat input (W) - None for calculated value
    Q_roof=0,         # Roof heat ingress (W)
    eta_w=0.70,       # Wall heat partitioning fraction (-)
    k_w=0.04,         # Wall thermal conductivity (W/mK)
    rho_w=60,         # Wall density (kg/m³)
    cp_w=900,         # Wall specific heat capacity (J/kgK)
    h_L=135,          # Liquid heat transfer coefficient (W/m²K)
    T_init=True       # Initialize wall temperature with ambient
)

# Step 4: Configure environmental temperature variations
tank.set_EnvironmentalProps(
    T_avg_day=278.15,      # Average daily ambient temperature (K)
    T_range_day=15,        # Amplitude of daily temperature fluctuation (K)
    h_env=14,              # Environmental heat transfer coefficient (W/m²K)
    p_anual=None           # Annual temperature variation coefficients (set to None for daily cycle only)
)

# Step 5: Run evaporation simulation
tank.evaporate(3600*24)  # Simulate for 24 hours (1 day)

# Step 6: Visualize results
tank.plot_tv()  # Plot temperature vs time
```

## Examples

Check the `notebooks/` folder for detailed examples and tutorials.
