import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_enzyme(matrix, label):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(matrix, edgecolor='k')
    fig.canvas.set_window_title(label)

def volume_plot(data):
    #seperate out into primary structural channels
    structures = {"cat":data[:, :, :, 0],
                  "bind":data[:, :, :, 1],
                  "het":data[:, :, :, 2],
                  "struct":data[:, :, :, 3]}
    total_enzyme = np.concatenate([structures[x] for x in structures])
    for structure in structures:
        plot_enzyme(structures[structure], structure)
    plot_enzyme(total_enzyme, 'total')
    plt.show()

voxel = np.load('/Users/kenny/desktop/chem195/enzyme_classifier/Xs_generator/Xs/3ZOK_X.npy')
volume_plot(voxel)
