'''
train model on dataset "porosity_ran"
'''
import torch
import numpy as np
from torch.optim import SGD
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
import os
from torch import nn

from LoadData import *

save_result = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# train = datasets.MNIST('data/', download=True, train=True)
# test = datasets.MNIST('data/', download=True, train=False)

# training console
train_batch_size=16
val_batch_size=32
root_dir=os.getcwd()
model_index='mymodel'
data_index='mymodel'
categorized_data=False

x_train,y_train,x_val,y_val=load_data(categorized_data,data_index,model_index)
mydataset = TensorDataset(x_train, y_train)
trainloader = DataLoader(mydataset, batch_size=train_batch_size, shuffle=True)

class BN_Conv3d(nn.Module):
    """
    BN_CONV_RELU
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, padding: object,
                 dilation=1, groups=1, bias=False) -> object:
        super(BN_Conv3d, self).__init__()
        self.seq = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding,padding_mode='circular', dilation=dilation, groups=groups, bias=bias),
            )

    def forward(self, x):
        return self.seq(x)


class DenseBlock3d(nn.Module):
    '''number of output channel = in_channel+n_layer*growth_rate'''

    def __init__(self, input_channels, num_layers, growth_rate, kernalsize=3):
        super(DenseBlock3d, self).__init__()
        self.num_layers = num_layers
        self.k0 = input_channels
        self.k = growth_rate
        self.kernalsize=kernalsize
        self.layers = self.__make_layers()

    def __make_layers(self):
        layer_list = nn.Sequential()
        for i in range(self.num_layers):
            layer_list.add_module(name=str(i),module=nn.Sequential(
                BN_Conv3d(self.k0 + i * self.k, 4 * self.k, 1, 1, 0),
                BN_Conv3d(4 * self.k, self.k, self.kernalsize, 1, padding=self.kernalsize//2)
            ))
        return layer_list

    def forward(self, x):
        feature = self.layers[0](x)
        out = torch.cat((x, feature), 1)
        for i in range(1, len(self.layers)):
            feature = self.layers[i](out)
            out = torch.cat((feature, out), 1)
        return out

class myDenseNet3d(nn.Module):

    def __init__(self, layers: object, kernalsizes: object, k, theta, num_classes) -> object:
        super(myDenseNet3d, self).__init__()
        # params
        self.layers = layers
        self.kernalsizes = kernalsizes
        self.k = k
        self.theta = theta
        self.num_classes = num_classes
        # layers
        self.conv = self.BN_Conv3d0(1, 2 * k, 3, 1, 1)
        self.blocks, patches = self.__make_blocks(2 * k)
        self.pdrop=0.2
        self.fc =  nn.Sequential(
            nn.Linear(in_features=patches, out_features=128, bias=True),
            nn.Dropout(self.pdrop),
            nn.Tanh(),
            nn.Linear(in_features=128, out_features=32, bias=True),
            nn.Dropout(self.pdrop),
            nn.Tanh(),
            nn.Linear(in_features=32, out_features=8, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=8, out_features=num_classes, bias=True),
        )

    def BN_Conv3d0(self, in_channels: object, out_channels: object, kernel_size: object, stride: object,
                  padding: object, dilation=1, groups=1, bias=False):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation,padding_mode='circular', groups=groups, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def __make_transition(self, in_chls):
        '''reduce channel numbers and feature map dimension through 1x1 conv and pooling'''
        out_chls = int(self.theta * in_chls)
        return nn.Sequential(
            BN_Conv3d(in_chls, out_chls, 1, 1, 0),
            nn.AvgPool3d(2)
        ), out_chls

    def __make_blocks(self, k0):
        """
        make block-transition structures
        :param k0:
        :return:
        """
        # layers_list = []
        layers_list=nn.Sequential()
        patches = 0

        for i in range(len(self.layers)):
            layers_list.add_module('DenseBlock3d'+str(i),DenseBlock3d(k0, self.layers[i], self.k, kernalsize=self.kernalsizes[i]))
            patches = k0 + self.layers[i] * self.k  # output feature patches from Dense Block
            if i != len(self.layers) - 1:
                transition, k0 = self.__make_transition(patches)
                layers_list.add_module('transition'+str(i),transition)
        return layers_list, patches  # *layers_list=layers_list[0],layers_list[1],...

    def forward(self, x):
        out = self.conv(x)
        # out = F.max_pool3d(out, 3, 1, 1)
        # print(out.shape)
        out = self.blocks(out)
        # print(out.shape)
        out = F.adaptive_avg_pool3d(out, 1)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = myDenseNet3d([8, 16, 10], [3,3,3], k=4, theta=0.5, num_classes=1).cuda(device)

opt = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,0.998)

# ----------- self-defined MSE loss ------------------
def mymseloss(xy, z):
    valloader = DataLoader(TensorDataset(xy, z), batch_size=val_batch_size, shuffle=False)
    with torch.no_grad():
        err_z = []
        first_round = True
        for xy1, z1 in valloader:
            if first_round:
                err_z = z1 - model(xy1).data
                first_round = False
            else:
                err_z = torch.vstack([err_z, z1 - model(xy1)])
        sqr_err1 = torch.square(err_z)
        mean_sqr_err1 = torch.mean(sqr_err1)
        torch.cuda.empty_cache()
    return mean_sqr_err1

loss_function = nn.MSELoss().cuda(device)

trainlog = []
vallog = []
model.train()
for epoch in range(0,1000):
    t0=time.time()
    for xy, z in trainloader:
        z_pr = model(xy)
        loss = loss_function(z_pr, z)
        opt.zero_grad()
        loss.backward()
        opt.step()
    save_file = os.path.join(root_dir,'modellog/model'+model_index+'_epoch' + str(epoch) + '.pt')
    torch.save(model, save_file)
    torch.cuda.empty_cache()
    print(epoch,'train_time: ',time.time()-t0)

    if epoch % 5 == 0:
        # calculate val loss
        model.eval()

        with torch.no_grad():
            valloss = mymseloss(x_val, y_val)
            trainloss = mymseloss(x_train, y_train)
            trainlog.append(np.sqrt(trainloss.item()))
            vallog.append(np.sqrt(valloss.item()))
        print(epoch, 'train loss:', np.sqrt(trainloss.item()), 'val loss:', np.sqrt(valloss.item()))
        with open('./RMSEloss_evolution.txt', 'a') as f:
            f.writelines(str(epoch) + ' ' + str(np.sqrt(trainloss.item())) + ' ' + str(np.sqrt(valloss.item())) + '\n')
        save_file = os.path.join(root_dir,'modellog/model'+model_index+'_epoch'+str(epoch)+'trainloss_'+str(np.sqrt(trainloss.item()))+'valloss_'+str(np.sqrt(valloss.item()))+'.pt')
        torch.save(model,save_file)
        model.train()


