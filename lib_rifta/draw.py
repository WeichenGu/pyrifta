import numpy as np
import matplotlib.pyplot as plt

def draw_init(X, Y, Z, selection, xx, yy, Z_avg, brf_params, ext_name, map_name):
    # map_height = 1 + np.count_nonzero(selection)
    # map_name = ['height_error', 'dwell_time', 'residual', 'removal']
    map_height = 1 + np.count_nonzero(selection)
    grid = plt.GridSpec(map_height, 4)
    fig = plt.figure("One-step surface extension dwell time results", figsize=(16, 10), dpi=800)

    # Original surface
    ax0 = fig.add_subplot(grid[0, 0:2])
    mesh0 = ax0.pcolormesh(X * 1e3, Y * 1e3, Z * 1e9, cmap='viridis')
    ax0.set_aspect('equal')
    ax0.invert_yaxis()
    c0 = plt.colorbar(mesh0, ax=ax0, pad=0.05)
    c0.set_label('[nm]')
    rms_Z = np.std(Z) * 1e9
    ax0.set_title(f"Original Surface: PV = {np.round((np.max(Z) - np.min(Z)) * 1e9, 2)} nm, RMS = {np.round(rms_Z, 2)} nm")

    # BRF
    ax0 = fig.add_subplot(grid[0, 2:3])
    mesh0 = ax0.pcolormesh(xx * 1e3, yy * 1e3, Z_avg * 1e9, cmap='viridis')
    ax0.set_aspect('equal')
    c0 = plt.colorbar(mesh0, ax=ax0, pad=0.05)
    c0.set_label('[nm]')
    ax0.set_title(f"BRF: Peakrate = {np.round(brf_params['A']*1e9, 3)} nm/s, Sigma = {np.round(brf_params['sigma_xy'][0] * 1e3, 4)} nm")

    fig.tight_layout()
    fig.subplots_adjust(top=0.99, bottom=0.1, left=0.1, right=0.8, hspace=0.5, wspace=0.4)
    
    return fig, grid, ax0, mesh0




def draw_simulation(fig, X, Y, X_ext, Y_ext, Z_ext, T_ext, Z_residual_ca, Z_removal_dw,
                    Grid, ext_name, map_name, map_row):
    # grid = plt.GridSpec(map_height, 4)
    ax1 = fig.add_subplot(Grid[map_row,0])
    mesh1 = ax1.pcolormesh(X_ext * 1e3, Y_ext * 1e3, Z_ext * 1e9, cmap='viridis')
    ax1.set_aspect('equal')
    c1 = plt.colorbar(mesh1, ax=ax1, pad=0.05)
    c1.set_label('[nm]')
    ax1.set_title(f"{ext_name[map_row-1]}: \nPV = {np.round((np.max(Z_ext) - np.min(Z_ext)) * 1e9, 2)} nm, RMS = {np.round(np.std(Z_ext) * 1e9, 2)} nm")

    ax2 = fig.add_subplot(Grid[map_row,1])
    mesh2 = ax2.pcolormesh(X_ext * 1e3, Y_ext * 1e3, T_ext, cmap='viridis')
    ax2.set_aspect('equal')
    c2 = plt.colorbar(mesh2, ax=ax2, pad=0.05)
    c2.set_label('[s]')
    ax2.set_title(f"{ext_name[map_row-1]}: \ndwell time =  {np.round((np.sum(T_ext)), 2)} s")

    ax3 = fig.add_subplot(Grid[map_row,2])
    mesh3 = ax3.pcolormesh(X * 1e3, Y * 1e3, Z_residual_ca * 1e9, cmap='viridis')
    ax3.set_aspect('equal')
    c3 = plt.colorbar(mesh3, ax=ax3, pad=0.05)
    c3.set_label('[nm]')
    ax3.set_title(f"{ext_name[map_row-1]}: residual \nPV =  {np.round((np.max(Z_residual_ca) - np.min(Z_residual_ca)) * 1e9, 2)} nm, RMS = {np.round(np.std(Z_residual_ca) * 1e9, 2)} nm")
    ax3.set_xlim(ax2.get_xlim())
    ax3.set_ylim(ax2.get_ylim())

    ax4 = fig.add_subplot(Grid[map_row,3])
    mesh4 = ax4.pcolormesh(X_ext * 1e3, Y_ext * 1e3, Z_removal_dw * 1e9, cmap='viridis')
    ax4.set_aspect('equal')
    c4 = plt.colorbar(mesh4, ax=ax4, pad=0.05)
    c4.set_label('[nm]')
    ax4.set_title(f"{ext_name[map_row-1]}: removal \nPV =  {np.round((np.max(Z_removal_dw) - np.min(Z_removal_dw)) * 1e9, 2)} nm, RMS = {np.round(np.std(Z_removal_dw) * 1e9, 2)} nm")

    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax4.invert_yaxis()
    fig.tight_layout()


