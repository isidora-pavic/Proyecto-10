import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fontsize_label  = 14
fontsize_ticks  = 14
fontsize_legend = 14

# Read CSV file
data_LF95_df = pd.read_csv('Data/large_tank_data_LF95_3weeks.csv')

# Convert DataFrame to dictionary to maintain the same syntax
data_LF95 = data_LF95_df.to_dict('series')

# Normalized heat transfer rates visualization
cmap = plt.get_cmap('inferno', 6)

def normalize(arr):
    return arr / np.max(np.abs(arr))

# Mask to select data starting from a specific time
time_from = 20
time_to   = 200
mask = data_LF95['Time'] >= time_from*3600

# Create figure
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(data_LF95['Time'][mask]/3600, normalize(data_LF95['Q_tot'][mask]), label=r'$\dot{Q}_{\text{tot}}$', color=cmap(0))
plt.plot(data_LF95['Time'][mask]/3600, normalize(data_LF95['Q_L'][mask]), label=r'$\dot{Q}_{\text{Li}}$', color=cmap(4))
plt.plot(data_LF95['Time'][mask]/3600, normalize(data_LF95['Q_b'][mask]), label=r'$\dot{Q}_{\text{b}}$', color=cmap(3))
plt.plot(data_LF95['Time'][mask]/3600, normalize(data_LF95['Q_Vw'][mask]), label=r'$\dot{Q}_{\text{Wi}}$', color=cmap(2))

plt.xlim(time_from,time_to)
plt.ylabel(r'Normalized Heat Transfer Rate', fontsize=fontsize_label)
plt.xlabel('Time / h', fontsize=fontsize_label)
plt.tick_params(labelsize=fontsize_ticks)
plt.legend(fontsize=fontsize_legend, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
plt.tight_layout()

plt.savefig('Figures/Fig_8.svg', dpi=300, bbox_inches='tight')