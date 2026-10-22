import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fontsize_label  = 14
fontsize_ticks  = 14
fontsize_legend = 14

# Read CSV files
data_LF50_df = pd.read_csv('Data/large_tank_data_LF50_3weeks.csv')
data_LF95_df = pd.read_csv('Data/large_tank_data_LF95_3weeks.csv')

# Extract T_w_raw columns for LF95 and LF50
tw_cols_LF95 = [col for col in data_LF95_df.columns if col.startswith('T_w_raw_')]
tw_cols_LF50 = [col for col in data_LF50_df.columns if col.startswith('T_w_raw_')]

# Create T_w_raw arrays (transpose to match original structure: rows=radial nodes, cols=time)
T_w_raw_LF95 = data_LF95_df[tw_cols_LF95].values.T
T_w_raw_LF50 = data_LF50_df[tw_cols_LF50].values.T

# Convert DataFrames to dictionaries to maintain the same syntax
data_LF95 = data_LF95_df.to_dict('series')
data_LF50 = data_LF50_df.to_dict('series')

# Add T_w_raw arrays to dictionaries
data_LF95['T_w_raw'] = T_w_raw_LF95
data_LF50['T_w_raw'] = T_w_raw_LF50

cmap = plt.get_cmap('inferno', 7)
T_env_avg = 5.3286+273.15     # K
T_range   = 15                # K

T_env = lambda t: T_env_avg + 0.5*T_range * np.sin(t * 2*np.pi/(24*3600))

fig, axs =  plt.subplots(1, 2, figsize=(15, 6), dpi=200)

# LF95 subplot
axs[0].plot(data_LF95['Time']/3600, T_env(data_LF95['Time']), label=r'$T_{\text{air}}$', color=cmap(0))
axs[0].plot(data_LF95['Time']/3600, data_LF95['Tw_avg'], label=r'$T_{\text{W,avg}}$', color=cmap(2))
axs[0].plot(data_LF95['Time']/3600, data_LF95['T_BOG'], label=r'$T_{\text{BOG}}$', color=cmap(3))
axs[0].plot(data_LF95['Time']/3600, data_LF95['Tv_avg'], label=r'$T_{\text{V,avg}}$', color=cmap(4))
axs[0].plot(data_LF95['Time'][2:]/3600, data_LF95['T_w_raw'][0,2:], label=r'$T_{\text{W},\text{r}=0}$', color=cmap(5))

axs[0].set_xlabel('Time / h', fontsize=fontsize_label)
axs[0].set_xlim(-1, 500)
axs[0].set_ylabel(r'Temperature / K', fontsize=fontsize_label)
axs[0].tick_params(axis='both', labelsize=fontsize_ticks)
axs[0].legend(loc=(0.52, 0.13), fontsize=fontsize_legend, ncol=2)

# LF50 subplot
axs[1].plot(data_LF50['Time']/3600, T_env(data_LF50['Time']), label=r'$T_{\text{air}}$', color=cmap(0))
axs[1].plot(data_LF50['Time']/3600, data_LF50['Tw_avg'], label=r'$T_{\text{W,avg}}$', color=cmap(2))
axs[1].plot(data_LF50['Time']/3600, data_LF50['T_BOG'], label=r'$T_{\text{BOG}}$', color=cmap(3))
axs[1].plot(data_LF50['Time']/3600, data_LF50['Tv_avg'], label=r'$T_{\text{V,avg}}$', color=cmap(4))
axs[1].plot(data_LF50['Time'][2:]/3600, data_LF50['T_w_raw'][0,2:], label=r'$T_{\text{W},\text{r}=0}$', color=cmap(5))

axs[1].set_xlabel('Time / h', fontsize=fontsize_label)
axs[1].set_xlim(-1, 500)
axs[1].set_ylabel(r'Temperature / K', fontsize=fontsize_label)
axs[1].tick_params(axis='both', labelsize=fontsize_ticks)
axs[1].legend(loc=(0.52, 0.13), fontsize=fontsize_legend, ncol=2)

plt.tight_layout()
plt.savefig('Figures/Fig_6.svg', dpi=300, bbox_inches='tight')