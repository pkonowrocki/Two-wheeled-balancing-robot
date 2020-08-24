import sys
sys.path.append('../')
from torchdyn.models import *
from torchdyn import *
from torchdyn.datasets import *

d = ToyDataset()
X, yn = d.generate(n_samples=2048, dataset_type='spirals', noise=.4)

import matplotlib.pyplot as plt

colors = ['orange', 'blue']
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
for i in range(len(X)):
    ax.scatter(X[i,0], X[i,1], color=colors[yn[i].int()])

import torch
import torch.utils.data as data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X_train = torch.Tensor(X).to(device)
y_train = torch.LongTensor(yn.long()).to(device)
train = data.TensorDataset(X_train, y_train)
trainloader = data.DataLoader(train, batch_size=len(X), shuffle=True)


import torch.nn as nn
import pytorch_lightning as pl

class Learner(pl.LightningModule):
    def __init__(self, model:nn.Module):
        super().__init__()
        self.model = model
        self.c = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_dataloader(self):
        return trainloader


# vector field parametrized by a NN
f = nn.Sequential(
        nn.Linear(4, 64),
        nn.Tanh(),
        nn.Linear(64, 2))

# Neural ODE
model = NeuralDE(f,
                 order=2,
                 solver='dopri5',
                 sensitivity='adjoint').to(device)

seq = nn.Sequential(Augmenter(1, 2, order='last'), model, nn.Linear(4, 2)).to(device)

learn = Learner(seq)
trainer = pl.Trainer(min_epochs=10, max_epochs=15)
trainer.fit(learn)

# Evaluate the data trajectories
s_span = torch.linspace(0,1,100)
data = X_train[::10,:]
X_d = seq[0](data)
y_d = model(X_d)
trajectory = model.trajectory(X_d, s_span).detach().cpu()

# Trajectories in the depth domain
plot_2D_depth_trajectory(s_span, trajectory[:,:,:2], yn[::10], len(X)//10)
plot_2D_depth_trajectory(s_span, trajectory[:,:,2:4], yn[::10], len(X)//10)

# Trajectories in the state-space
plot_2D_state_space(trajectory[:,:,-2:], yn[::10], len(X)//10)