import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os


def load_data(categorized_data=True, data_index=None, model_index=None,
              device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    root_dir = os.getcwd()
    if categorized_data:
        """need to supply 4 ".pt" files"""
        x_train = torch.load(os.path.join(root_dir, 'x_train_for_model' + data_index + '.pt'))
        y_train = torch.load(os.path.join(root_dir, 'y_train_for_model' + data_index + '.pt'))
        x_val = torch.load(os.path.join(root_dir, 'x_val_for_model' + data_index + '.pt'))
        y_val = torch.load(os.path.join(root_dir, 'y_val_for_model' + data_index + '.pt'))
    else:
        """read raw data, generate and save .pt file for PyTorch"""
        save_result = True

        def data_trans(posi, delt_l=0.05):
            return (posi[:, 0:3] // delt_l).astype(int)

        fracturestrain_real = torch.from_numpy(
            np.loadtxt(os.path.join(root_dir, 'fractureStrain_ran.txt')).astype(np.float32))
        figdata_real = torch.from_numpy(np.load(os.path.join(root_dir, 'porosdata10000ran.npy')).astype(np.float32))
        # ================= data augmentation ================
        expansion = np.loadtxt(os.path.join(root_dir, 'expansion_ran.txt')).astype(int)
        figdata_real_aug = torch.zeros([expansion.sum(), 1, 20, 20, 20])
        fracturestrain_real_aug = torch.zeros([expansion.sum()])
        figdata_real_aug[:len(figdata_real), :, :, :, :] = figdata_real
        fracturestrain_real_aug[:len(fracturestrain_real)] = fracturestrain_real
        n = len(fracturestrain_real)
        for i in range(len(expansion)):
            for j in range(expansion[i] - 1):
                # print(i)
                figdata_real_aug[n, :, :, :, :] = torch.roll(
                    torch.roll(torch.roll(figdata_real[i:i + 1, 0:1, :, :, :], np.random.randint(1, 19), dims=2),
                               np.random.randint(1, 19), dims=3), np.random.randint(1, 19), dims=4)
                fracturestrain_real_aug[n] = fracturestrain_real[i]
                n = n + 1
        figdata_real = figdata_real_aug
        fracturestrain_real = fracturestrain_real_aug
        # ====================data argmentation end=================
        fracturestrain = (fracturestrain_real - fracturestrain_real.mean()) / fracturestrain_real.std()
        figdata = (figdata_real - figdata_real.mean()) / figdata_real.std()

        a = np.random.rand(int(len(figdata) * 0.2)) * len(figdata)
        val_index = []
        for i in a:
            val_index.append(int(i))
        val_index = np.unique(val_index).tolist()

        train_index = []
        for i in range(0, len(figdata)):
            if (i not in val_index):
                train_index.append(i)

        x_train = figdata[train_index].cuda(device)
        y_train = fracturestrain[train_index].unsqueeze(1).cuda(device)
        x_val = figdata[val_index].cuda(device)
        y_val = fracturestrain[val_index].unsqueeze(1).cuda(device)

        torch.save(x_train, os.path.join(root_dir, 'x_train_for_model' + model_index + '.pt'))
        torch.save(y_train, os.path.join(root_dir, 'y_train_for_model' + model_index + '.pt'))
        torch.save(x_val, os.path.join(root_dir, 'x_val_for_model' + model_index + '.pt'))
        torch.save(y_val, os.path.join(root_dir, 'y_val_for_model' + model_index + '.pt'))

    return x_train,y_train,x_val,y_val

