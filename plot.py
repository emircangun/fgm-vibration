import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import config
from scipy.interpolate import griddata

from model import *
from train_and_evaluate import *


# turbo_cmap = plt.get_cmap("turbo")
# colors = [turbo_cmap(i) for i in np.linspace(0, 1, 256)]
# red_extension = [turbo_cmap(1) for _ in range(64)]
# colors += red_extension
# custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_turbo", colors, N=len(colors))


def plot_losses(train_losses, eval_losses):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(eval_losses, label="Evaluation Loss")
    plt.xlabel("# epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig(config.LOSS_FILE)

    print(f"Loss file saved as {config.LOSS_FILE}!")
    
    plt.close()

def plot_results_3d():    
    fig, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')


    X = [5, 20]
    Y = np.arange(0, 5, 1)

    xx, yy = np.meshgrid(X, Y)
    pairs = np.column_stack((xx.reshape(-1), yy.reshape(-1)))

    # Z -> model output
    model_file_path = config.MODEL_WEIGHTS_FILE
    model = load_model(model_file_path)

    # first input
    # inp = [["CBT", "S-S", l_h, k_value] for l_h, k_value in pairs]
    # out = test_examples(model, inp).detach().numpy()

    # second input
    inp2 = [["HSDBT", "S-S", l_h, k_value] for l_h, k_value in pairs]
    out2 = test_examples(model, inp2).detach().numpy()
    # out2 = out2 / out

    grid_x, grid_y = np.mgrid[5:20:200j, 0:5:200j]
    # interpolated_grid1 = griddata(pairs, out, (grid_x, grid_y), method='cubic')
    # interpolated_grid1 = np.array(interpolated_grid1).squeeze()

    interpolated_grid2 = griddata(pairs, out2, (grid_x, grid_y), method='cubic')
    interpolated_grid2 = np.array(interpolated_grid2).squeeze()

    ##################################################################################################
    
    # vmin = min(out.min(), out2.min())
    # vmax = max(out.max(), out2.max())
    vmin = out2.min()
    vmax = out2.max()

    # surf1 = ax1.plot_surface(grid_x,
    #                        grid_y,
    #                        interpolated_grid1,
    #                        cmap=cm.turbo,
    #                        linewidth=0, antialiased=False,
    #                        rcount=300, ccount=300,
    #                        vmin=vmin, vmax=vmax)
    
    surf2 = ax2.plot_surface(grid_x,
                           grid_y,
                           interpolated_grid2,
                           cmap=cm.turbo,
                           linewidth=0, antialiased=False,
                           rcount=300, ccount=300,
                           vmin=vmin, vmax=vmax)



    # fig.colorbar(surf1, shrink=0.7, aspect=15)
    fig.colorbar(surf2, shrink=0.7, aspect=15)

    # ax1.zaxis.set_major_formatter('{x:.02f}')
    # ax1.set_xlabel('L/h')
    # ax1.set_ylabel('p')
    # ax1.set_zlabel('ANN Prediction')
    # ax1.invert_xaxis()
    # ax1.set_xticks([0, 5, 10, 15, 20])
    # ax1.set_yticks(np.arange(0, 5, 1))
    # ax1.view_init(azim=45, elev=25)

    ax2.zaxis.set_major_formatter('{x:.02f}')
    ax2.set_xlabel('L/h')
    ax2.set_ylabel('p')
    ax2.set_zlabel('Present HSDT')
    ax2.invert_xaxis()
    ax2.w_zaxis.line.set_lw(0.)
    ax2.set_zticks([])
    ax2.set_xticks([0, 5, 10, 15, 20])
    ax2.set_yticks(np.arange(0, 5, 1))
    # ax2.view_init(azim=45, elev=25)
    ax2.view_init(azim=0, elev=90)

    plt.show()