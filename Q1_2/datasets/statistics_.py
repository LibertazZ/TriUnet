from datasets.dataloader import DataSet_for_one_to_statistics
import numpy as np

dataset = DataSet_for_one_to_statistics()

z_min, z_max = float('inf'), float('-inf')
zdr_min, zdr_max = float('inf'), float('-inf')
kdp_min, kdp_max = float('inf'), float('-inf')

for i in range(len(dataset)):
    print(i)
    zin, zdr, kdp, zout = dataset[i]
    # print(np.sum(zin))
    zmin = min(np.min(zin), np.min(zout))
    zmax = max(np.max(zin), np.max(zout))
    if zmin < z_min:
        z_min = zmin
    if zmax > z_max:
        z_max = zmax

    zdrmin, zdrmax = np.min(zdr), np.max(zdr)
    if zdrmin < zdr_min:
        zdr_min = zdrmin
    if zdrmax > zdr_max:
        zdr_max = zdrmax
    kdpmin, kdpmax = np.min(kdp), np.max(kdp)
    if kdpmin < kdp_min:
        kdp_min = kdpmin
    if kdpmax > kdp_max:
        kdp_max = kdpmax
    # break

print(z_min, z_max, zdr_min, zdr_max, kdp_min, kdp_max)