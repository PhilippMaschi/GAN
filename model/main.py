from torch import nn, optim, float32, full, randn
from torch.utils.data import DataLoader
import pandas as pd
import os
from tqdm import tqdm
import torch
import numpy as np
from math import ceil

from model.data_manip import data_prep_wrapper, invert_min_max_scaler, revert_reshape_arr
from model.layers import layersGen, layersDis
from model.plots import plot_losses


DAY_COUNT = 368
FEATURE_RANGE = (-1, 1)


class Generator(nn.Module):
    def __init__(self, model):
        super(Generator, self).__init__()
        self.model = model
    
    def forward(self, noise):
        output = self.model(noise)
        return output


class Discriminator(nn.Module):
    def __init__(self, model):
        super(Discriminator, self).__init__()
        self.model = model
    
    def forward(self, data):
        output = self.model(data).flatten()
        return output


class GAN(nn.Module):
    def __init__(
            self,
            dataset,
            outputPath,
            params,
            wandb = None,
            modelStatePath = None
        ):
        super().__init__()
        self.dataset, self.dfIdx, self.arr_minMax = \
            data_prep_wrapper(df = dataset, dayCount = DAY_COUNT, featureRange = FEATURE_RANGE)
        self.outputPath = outputPath
        self.wandb = wandb
        self.modelStatePath = modelStatePath
        # Get parameters from `params.py`
        for key, value in params.items():
            setattr(self, key, value)
        # Get layers from `layers.py`
        self.layersGen, self.layersDis = layersGen, layersDis

        self.dataLoader = \
            DataLoader(dataset = self.dataset, batch_size = self.batchSize, shuffle = True)
        self.Gen = Generator(model = self.layersGen).to(device = self.device)
        self.Dis = Discriminator(model = self.layersDis).to(device = self.device)
        self.optimGen = optim.Adam(params = self.Gen.parameters(), lr = self.lrGen, betas = self.betas)
        self.optimDis = optim.Adam(params = self.Dis.parameters(), lr = self.lrDis, betas = self.betas)

        self.df_loss = pd.DataFrame(columns = ['epoch', 'batch_index', 'loss_discriminator_real', 'loss_discriminator_fake', 'loss_generator'])
        self.epochSamples = []
        self.modelPath = self.outputPath / 'models'
        os.makedirs(self.modelPath)
        self.plotPath = self.outputPath / 'plots'
        os.makedirs(self.plotPath)

        # Continue training model
        if self.modelStatePath:
            self.modelState = torch.load(self.modelStatePath)
            if not (self.dfIdx == self.modelState['dfIdx']).all():
                raise ValueError('Timestamps do not match!')
            self.arr_minMaxOld = self.modelState['minMax']
            self.arr_minMax = np.array([min(self.arr_minMax[0], self.arr_minMaxOld[0]), max(self.arr_minMax[1], self.arr_minMaxOld[1])])
            self.Gen.load_state_dict(self.modelState['gen_state_dict'])
            self.Dis.load_state_dict(self.modelState['dis_state_dict'])
            self.optimGen.load_state_dict(self.modelState['optim_gen_state_dict'])
            self.optimDis.load_state_dict(self.modelState['optim_dis_state_dict'])


    def train(self, progress = None, root = None):
        gen_loss_history = []
        learning_rate_halfed = []
        for epoch in tqdm(range(self.epochCount)):
            totalLossGen, totalLossDisFake, totalLossDisReal = 0, 0, 0
            for batchIdx, data in enumerate(self.dataLoader):
                xReal = data.to(device = self.device, dtype = float32)
                labelsReal = \
                    full(size = (xReal.shape[0],), fill_value = self.labelReal, dtype = float32, device = self.device)
                labelsFake = \
                    full(size = (xReal.shape[0],), fill_value = self.labelFake, dtype = float32, device = self.device)

                # Train discriminator with real data
                self.Dis.zero_grad()
                yReal = self.Dis(xReal)
                lossDisReal = self.lossFct(yReal, labelsReal)
                lossDisReal.backward()

                # Train discriminator with fake data
                noise = randn(xReal.shape[0], self.dimNoise, 1, 1, device = self.device)
                xFake = self.Gen(noise)
                yFake = self.Dis(xFake.detach())
                lossDisFake = self.lossFct(yFake, labelsFake)
                lossDisFake.backward()
                self.optimDis.step()

                # Train generator
                for idx in range(self.loopCountGen):
                    self.Gen.zero_grad()
                    xFake = self.Gen(noise)
                    yFakeNew = self.Dis(xFake)
                    lossGen = self.lossFct(yFakeNew, labelsReal).clone()
                    lossGen.backward(retain_graph = True if idx < self.loopCountGen - 1 else False)
                    self.optimGen.step()

                # Update total loss
                totalLossGen += lossGen.cpu().item()
                totalLossDisFake += lossDisFake.cpu().item()
                totalLossDisReal += lossDisReal.cpu().item()

                # Log progress
                self.logger(epoch, batchIdx, lossDisReal, lossDisFake, lossGen)

            # learning rate decay:
            mean_gen_loss = totalLossGen / len(self.dataLoader)
            gen_loss_history.append(mean_gen_loss)

            # if len(gen_loss_history) > 30 and epoch > 450: # start decay earliest after 200 epochs
            if epoch == 500 or epoch == 1000:
                # recent_losses = gen_loss_history[-30:]
                # if mean_gen_loss >= min(recent_losses) and not any(learning_rate_halfed[-300:])==True:  # No improvement
                    # learning_rate_halfed.append(True)
                    for param_group in self.optimGen.param_groups:
                        param_group['lr'] *= 0.5  # Halve generator learning rate
                    for param_group in self.optimDis.param_groups:
                        param_group['lr'] *= 0.5  # Halve discriminator learning rate
                    print(f"Learning rate halved at epoch {epoch + 1}")
            #     else:
            #         learning_rate_halfed.append(False)
            # else:
            #     learning_rate_halfed.append(False)

            # Log progress with wandb
            if self.wandb:
                self.wandb.log({
                    'loss_discriminator_real': totalLossDisReal/len(self.dataLoader),
                    'loss_discriminator_fake': totalLossDisFake/len(self.dataLoader),
                    'loss_generator': totalLossGen/len(self.dataLoader),
                    # 'learning_rate_generator': param_group['lrDis']
                })

            # Save model state
            if (epoch + 1) % self.modelSaveFreq == 0 or epoch + 1 == self.epochCount:
                self.save_model_state(epoch)

            # Track progress (save generated samples)
            if self.trackProgress:
                self.epochSamples.append(self.generate_data())

            # Advance progress bar
            if progress:
                if (epoch + 1)%(ceil(self.epochCount/10)) == 0 or epoch + 1 == self.epochCount:
                    progress['value'] += 7
                    root.update()

        # Plot losses
        plot_losses(self.df_loss, self.plotPath)

        # Save losses in CSV file
        self.df_loss.to_csv(self.outputPath / 'losses.csv', index = False)

    def logger(self, epoch, batchIdx, lossDisReal, lossDisFake, lossGen):
        self.df_loss.loc[len(self.df_loss)] = epoch, batchIdx, lossDisReal.cpu().item(), lossDisFake.cpu().item(), lossGen.cpu().item()
    
    def save_model_state(self, epoch):
        torch.save({
            'device': self.device,
            'epoch': epoch,
            'dimNoise': self.dimNoise,
            'profileCount': self.dataset.shape[0],
            'dfIdx': self.dfIdx,
            'minMax': self.arr_minMax,
            'gen_layers': self.layersGen,
            'dis_layers': self.layersDis,
            'gen_state_dict': self.Gen.state_dict(),
            'dis_state_dict': self.Dis.state_dict(),
            'optim_gen_state_dict': self.optimGen.state_dict(),
            'optim_dis_state_dict': self.optimDis.state_dict(),
            'continued_from': self.modelStatePath,
            'feature_range': FEATURE_RANGE
        }, self.modelPath / f'epoch_{epoch + 1}.pt')

    def generate_data(self):
        noise = randn(self.dataset.shape[0], self.dimNoise, 1, 1, device = self.device)
        xSynth = self.Gen(noise)
        xSynth = xSynth.cpu().detach().numpy()
        xSynth = invert_min_max_scaler(xSynth, self.arr_minMax, FEATURE_RANGE)
        xSynth = revert_reshape_arr(xSynth)
        idx = self.dfIdx[:self.dfIdx.get_loc(0)]
        xSynth = xSynth[:len(idx)]
        xSynth = np.append(idx.to_numpy().reshape(-1, 1), xSynth, axis = 1)
        return xSynth


def generate_data_from_saved_model(modelStatePath):
    modelState = torch.load(modelStatePath)
    Gen = Generator(modelState['gen_layers'])
    Gen.load_state_dict(modelState['gen_state_dict'])
    noise = randn(modelState['profileCount'], modelState['dimNoise'], 1, 1, device = modelState['device'])
    xSynth = Gen(noise)
    xSynth = xSynth.cpu().detach().numpy()
    xSynth = invert_min_max_scaler(xSynth, modelState['minMax'], FEATURE_RANGE)
    xSynth = revert_reshape_arr(xSynth)
    idx = modelState['dfIdx'][:modelState['dfIdx'].get_loc(0)]
    xSynth = xSynth[:len(idx)]
    xSynth = np.append(idx.to_numpy().reshape(-1, 1), xSynth, axis = 1)
    return xSynth


def export_synthetic_data(arr, outputPath, fileFormat, filename = 'example_synth_profiles'):
    print(outputPath / f'{filename}')
    match fileFormat:
        case '.npy':
            np.save(file = outputPath / f'{filename}.npy', arr = arr)
        case '.csv':
            pd.DataFrame(arr).set_index(0).to_csv(outputPath / f'{filename}.csv')
        case '.xlsx':
            pd.DataFrame(arr).set_index(0).to_excel(outputPath / f'{filename}.xlsx')