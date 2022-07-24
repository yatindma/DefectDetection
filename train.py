import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split

device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
data = pd.read_csv("data.csv", sep=";")

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO
train, val = train_test_split(data, test_size=0.2, random_state=42)
# train, val = train_test_split(train, test_size=0.1, random_state=42)

dataloader_train = t.utils.data.DataLoader(ChallengeDataset(train, 'train'), batch_size=1, shuffle=False, num_workers=2)
dataloader_val = t.utils.data.DataLoader(ChallengeDataset(val, 'val'), batch_size=1, shuffle=False, num_workers=2)
# dataloader_test = t.utils.data.DataLoader(ChallengeDataset(val, 'val'), batch_size=1, shuffle=False, num_workers=2)

# create an instance of our ResNet model
# TODO
model = model.ResNet().to(device)

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO
criterion = t.nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
# Need to stop early here using the loss  - this functionality is already there in trainer.py


# go, go, go... call fit on trainer
# define the number of epochs
epochs = 15
trainer = Trainer(model=model,
                  crit=criterion,  # Loss function
                  optim=optimizer,  # Optimizer
                  train_dl=dataloader_train,  # Training data set
                  val_test_dl=dataloader_val,  # Validation (or test) data set
                  cuda=False,  # Whether to use the GPU
                  early_stopping_patience=-1)
res = 0#TODO
trainer.fit(epochs=epochs)
loss = trainer.val_test()
plt.plot(loss)
plt.show()

# plot the results
# plt.plot(np.arange(len(res[0])), res[0], label='train loss')
# plt.plot(np.arange(len(res[1])), res[1], label='val loss')
# plt.yscale('log')
# plt.legend()
# plt.savefig('losses.png')