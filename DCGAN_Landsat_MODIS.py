from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

import os
import numpy as np
import math
import glob
from LandsatModisDatasetLoader import LandsatMODISDataset
from Model import Discriminator, Generator

datapath = '/share/forest/MODIS_diffusion/Data/Dataset/modis_landsat_150x150/'  # Update the path accordingly
# Parameters
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 1e-4  # could also use two lrs, one for gen and one for disc
BATCH_SIZE = 64
IMAGE_SIZE = 150
CHANNELS_IMG = 5
NOISE_DIM = 100
NUM_EPOCHS = 50
FEATURES_DISC = 64
FEATURES_GEN = 64

   
# Create datasets and dataloaders
dataset_train = LandsatMODISDataset(datapath, vi='NDVI', mode='train', month='05', transform=lambda x: torch.tensor(x, dtype=torch.float32))
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

dataset_test = LandsatMODISDataset(datapath, vi='NDVI', mode='test', month='05',transform=lambda x: torch.tensor(x, dtype=torch.float32))
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(1, FEATURES_DISC).to(device)
initialize_weights(gen)
initialize_weights(disc)

# Optimizers
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
step = 0

gen.train()
disc.train()

# Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

#--------------------------
# Training
#--------------------------

from tqdm.notebook import tqdm
# Visualize result
import matplotlib.pyplot as plt
best_loss = float('inf')  # Initialize the best loss to infinity
for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    for batch_idx, (real, _) in enumerate(tqdm(dataloader_train)):
        _ = _.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)

        fake = gen(noise)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(_).reshape(-1)
        
        
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader_train)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

        if batch_idx % 50 == 0:  # Plot only for the first batch
            # Plot real images
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            for i, ax in enumerate(axes):
                ax.imshow(np.transpose(_[i].cpu().detach().numpy(), (1, 2, 0)))
                ax.axis('off')
                ax.set_title('Real')

            plt.show()

            # Plot fake images
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            for i, ax in enumerate(axes):
                ax.imshow(np.transpose(fake[i].cpu().detach().numpy(), (1, 2, 0)))
                ax.axis('off')
                ax.set_title('Fake')

            plt.show()

        # Save the best model
        if loss_disc < best_loss:
            best_loss = loss_disc
            torch.save(gen.state_dict(), "best_generator.pth")
            torch.save(disc.state_dict(), "best_discriminator.pth")
        step += 1