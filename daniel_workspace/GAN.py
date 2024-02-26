from torch import nn, optim, float32, full, randn
from torch.utils.data import DataLoader
import pandas as pd
import os
from tqdm import tqdm
import torch
import re
import numpy as np

from preproc import revert_reshape_arr, invert_min_max_scaler
from plots import plot_losses


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
            batchSize,
            modelGen,
            modelDis,
            lossFct,
            lrGen,
            lrDis,
            device,
            epochCount,
            labelReal,
            labelFake,
            dimNoise,
            outputPath,
            modelSaveFreq,
            wandb,
            betas,
        ):
        super().__init__()
        self.dataset = dataset
        self.batchSize = batchSize
        self.modelGen = modelGen
        self.modelDis = modelDis
        self.lossFct = lossFct
        self.lrGen = lrGen
        self.lrDis = lrDis
        self.device = device
        self.epochCount = epochCount
        self.labelReal = labelReal
        self.labelFake = labelFake
        self.dimNoise = dimNoise
        self.outputPath = outputPath
        self.modelSaveFreq = modelSaveFreq
        self.wandb = wandb
        self.betas = betas

        self.dataLoader = \
            DataLoader(dataset = self.dataset, batch_size = self.batchSize, shuffle = True) #NOTE: num_workers?
        self.Gen = Generator(model = self.modelGen).to(device = self.device)
        self.Dis = Discriminator(model = self.modelDis).to(device = self.device)
        self.lossFct = self.getLossFct()
        self.optimGen = optim.Adam(params = self.Gen.parameters(), lr = self.lrGen, betas = self.betas)
        self.optimDis = optim.Adam(params = self.Dis.parameters(), lr = self.lrDis, betas = self.betas)

        self.df_loss = pd.DataFrame(columns = ['epoch', 'batch_index', 'loss_discriminator_real', 'loss_discriminator_fake', 'loss_generator'])
        self.modelPath = self.outputPath / 'models'
        os.makedirs(self.modelPath)
        self.plotPath = self.outputPath / 'plots'
        os.makedirs(self.plotPath)

    def getLossFct(self):
        match self.lossFct:
            case 'L1':
                lossFct = nn.L1Loss()
            case 'MSE':
                lossFct = nn.MSELoss()
            case 'CrossEntropy':
                lossFct = nn.CrossEntropyLoss()
            case 'CTC':
                lossFct = nn.CTCLoss()
            case 'NLL':
                lossFct = nn.NLLLoss()
            case 'PoissonNLL':
                lossFct = nn.PoissonNLLLoss()
            case 'nn.GaussianNLL':
                lossFct = nn.GaussianNLLLoss()
            case 'KLDiv':
                lossFct = nn.KLDivLoss()
            case 'BCE':
                lossFct = nn.BCELoss()
            case 'BCEWithLogits':
                lossFct = nn.BCEWithLogitsLoss()
            case 'MarginRanking':
                lossFct = nn.MarginRankingLoss()
            case 'HingeEmbedding':
                lossFct = nn.HingeEmbeddingLoss()
        return lossFct

    def train(self):
        for epoch in tqdm(range(self.epochCount)):
            total_loss_Gen, total_loss_DisFake, total_loss_DisReal = 0, 0, 0
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
                noise = randn(xReal.shape[0], self.dimNoise, 1, 1, device = self.device)    #! generalization needed
                xFake = self.Gen(noise)
                yFake = self.Dis(xFake.detach())
                lossDisFake = self.lossFct(yFake, labelsFake)
                lossDisFake.backward()
                self.optimDis.step()

                # Train generator
                self.Gen.zero_grad()
                yFakeNew = self.Dis(xFake)
                lossGen = self.lossFct(yFakeNew, labelsReal)
                lossGen.backward()
                self.optimGen.step()

                total_loss_Gen += lossGen.cpu().item()
                total_loss_DisFake += lossDisFake.cpu().item()
                total_loss_DisReal += lossDisReal.cpu().item()

                # Log progress
                self.logger(epoch, batchIdx, lossDisReal, lossDisFake, lossGen)

            self.wandb.log({
                "lossGen": total_loss_Gen / len(self.dataLoader),
                "lossDisFake": total_loss_DisFake / len(self.dataLoader),
                "lossDisReal": total_loss_DisReal / len(self.dataLoader),
            })

            # Save model state
            if (epoch + 1) % self.modelSaveFreq == 0 or epoch + 1 == self.epochCount:
                self.save_model_state(epoch)
        
        # Plot losses
        plot_losses(self.df_loss, self.plotPath)

        # Save losses in CSV file
        self.df_loss.to_csv(self.outputPath / 'losses.csv', index = False)

    def logger(self, epoch, batchIdx, lossDisReal, lossDisFake, lossGen):
        self.df_loss.loc[len(self.df_loss)] = epoch, batchIdx, lossDisReal.cpu().item(), lossDisFake.cpu().item(), lossGen.cpu().item()
    
    def save_model_state(self, epoch):
        torch.save({
            'epoch': epoch,
            'gen_state_dict': self.Gen.state_dict(),
            'dis_state_dict': self.Dis.state_dict(),
            'optim_gen_state_dict': self.optimGen.state_dict(),
            'optim_dis_state_dict': self.optimDis.state_dict()
        }, self.modelPath / f'epoch_{epoch + 1}.pt')


def generate_data_from_saved_model(
        runPath,
        modelGen,
        device,
        profileCount,
        dimNoise,
        invertNorm = True
    ):
    file_dict = {
        int(re.findall(r'epoch_(\d+)\.pt', file)[-1]): file
        for file in os.listdir(runPath / 'models')
    }
    modelStatePath = runPath / 'models' / file_dict[max(file_dict)]
    minMax = np.load(runPath / 'min_max.npy')

    Gen = Generator(model = modelGen).to(device = device)
    modelState = torch.load(modelStatePath, map_location = torch.device(device))
    Gen.load_state_dict(modelState['gen_state_dict'])
    noise = randn(profileCount, dimNoise, 1, 1, device = device)    #! generalization needed
    xSynth = Gen(noise)
    xSynth = xSynth.cpu().detach().numpy()
    xSynth = revert_reshape_arr(xSynth)
    if invertNorm:
        xSynth = invert_min_max_scaler(xSynth, minMax)
    return xSynth