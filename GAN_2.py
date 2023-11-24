import torch
from torch import nn, cat, optim, full, randn, no_grad
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class Generator(nn.Module):
    def __init__(self, dimLatent, featureCount, classCount, dimEmbedding):
        super(Generator, self).__init__()
        self.dimLatent = dimLatent
        self.featureCount = featureCount
        self.classCount = classCount
        self.dimEmbedding = dimEmbedding    #dimension of the embedding tensor
        self.labelEmbedding = nn.Embedding(num_embeddings = self.classCount, embedding_dim = dimEmbedding)
        self.model = nn.Sequential(
            # 1st layer
            nn.Linear(in_features = self.dimLatent + self.dimEmbedding, out_features = 64),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # 2nd layer
            nn.Linear(in_features = 64, out_features = 128),
            #nn.ReLU(),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # 3rd layer
            nn.Linear(in_features = 128, out_features = featureCount),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        #print(labels, self.labelEmbedding(labels))
        x = self.model(cat((self.labelEmbedding(labels), noise), -1))   #apply model to concatenated tensor (fixed label tensor + noise tensor)
        return x


class Discriminator(nn.Module):
    def __init__(self, featureCount, classCount, dimEmbedding):
        super(Discriminator, self).__init__()
        self.featureCount = featureCount
        self.classCount = classCount
        self.dimEmbedding = dimEmbedding
        self.labelEmbedding = nn.Embedding(num_embeddings = self.classCount, embedding_dim = dimEmbedding)
        self.model = nn.Sequential(
            # 1st layer
            nn.Linear(in_features = self.featureCount + self.dimEmbedding, out_features = 128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # 2nd layer
            nn.Linear(in_features = 128, out_features = 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            # 3rd layer
            nn.Linear(in_features = 64, out_features = 1),
            nn.Sigmoid()
        )
    
    def forward(self, data, labels):
        bool_ = self.model(cat((data, self.labelEmbedding(labels)), -1))
        return bool_


class GAN(object):
    def __init__(self, name, device, dataLoader, dimLatent, featureCount, classCount, dimEmbedding, lr, maxNorm, epochCount, testLabel = None, exampleCount = 3):
        self.name = name
        self.device = device
        self.dataLoader = dataLoader
        self.dimLatent = dimLatent
        self.featureCount = featureCount
        self.classCount = classCount
        self.dimEmbedding = dimEmbedding
        self.lr = lr
        self.maxNorm = maxNorm
        self.epochCount = epochCount
        self.testLabel = testLabel
        self.exampleCount = exampleCount

        # Initialize generator
        self.Gen = Generator(dimLatent, featureCount, classCount, dimEmbedding)
        self.Gen.to(self.device)

        # Initialize discriminator
        self.Dis = Discriminator(featureCount, classCount, dimEmbedding)
        self.Dis.to(self.device)
    
        # Initialize optimizers
        self.optimGen = optim.Adam(params = self.Gen.parameters(), lr = self.lr)
        self.optimDis = optim.Adam(params = self.Dis.parameters(), lr = self.lr)

        # Initialize the loss function
        self.criterion = nn.BCELoss()

        self.df_loss = pd.DataFrame(
            columns = [
                'epoch',
                'batch index',
                'discriminator loss (real data)',
                'discriminator loss (fake data)',
                'discriminator loss',
                'generator loss',
                'discriminator gradient norm',
                'generator gradient norm'
            ])
        self.iterCount = 0
        if isinstance(self.testLabel, int):
            self.noiseFixed = randn(self.exampleCount, dimLatent, device = device)
            self.labelsFixed = full(size = (self.exampleCount,), fill_value = self.testLabel, device = self.device, dtype = torch.int32)
    
    def train(self):
        for epoch in tqdm(range(self.epochCount)):
            for batchIdx, (data, target) in enumerate(self.dataLoader): #target = actual (real) label
                data = data.to(device = self.device, dtype = torch.float32)
                target = target.to(device = self.device, dtype = torch.int32)

                # Train discriminator with real data
                self.Dis.zero_grad()                                                                                            #set the gradients to zero for every mini-batch
                yReal = self.Dis(data, target)                                                                                  #train discriminator with real data
                labelReal = full(size = (data.size(0), 1), fill_value = 1, device = self.device, dtype = torch.float32)         #a tensor containing only ones
                lossDisReal = self.criterion(yReal, labelReal)                                                                  #calculate the loss
                lossDisReal.backward()                                                                                          #calculate new gradients

                # Train discriminator with fake data
                noise = randn(data.size(0), self.dimLatent, device = self.device)                                               #create a tensor filled with random numbers
                randomLabelFake = torch.randint(low = 0, high = self.classCount, size = (data.size(0),), device = self.device)  #random labels needed in addition to the noise
                labelFake = full(size = (data.size(0), 1), fill_value = 0, device = self.device, dtype = torch.float32)         #a tensor containing only zeros
                xFake = self.Gen(noise, randomLabelFake)                                                                        #create fake data from noise + random labels with generator
                yFake = self.Dis(xFake.detach(), randomLabelFake)                                                               #let the discriminator label the fake data (`.detach()` creates a copy of the tensor)
                lossDisFake = self.criterion(yFake, labelFake)
                lossDisFake.backward()

                lossDis = (lossDisReal + lossDisFake)                                                                           #compute the total discriminator loss
                grad_norm_dis = torch.nn.utils.clip_grad_norm_(self.Dis.parameters(), max_norm = self.maxNorm)                  #gradient clipping (large max_norm to avoid actual clipping)
                self.optimDis.step()                                                                                            #update the discriminator

                # Train generator (now that we fed the discriminator with fake data)
                self.Gen.zero_grad()
                yFake_2 = self.Dis(xFake, randomLabelFake)                                                                      #let the discriminator label the fake data (now that the discriminator is updated)
                lossGen = self.criterion(yFake_2, labelReal)                                                                    #calculate the generator loss (small if the discriminator thinks that `yFake_2 == labelReal`)
                lossGen.backward()
                grad_norm_gen = torch.nn.utils.clip_grad_norm_(self.Gen.parameters(), max_norm = self.maxNorm)
                self.optimGen.step()

                # Log the progress
                self.df_loss.loc[len(self.df_loss)] = [
                    epoch,
                    batchIdx,
                    lossDisReal.detach().cpu().numpy(),
                    lossDisFake.detach().cpu().numpy(),
                    lossDis.detach().cpu().numpy(),
                    lossGen.detach().cpu().numpy(),
                    grad_norm_dis.detach().cpu().numpy(),
                    grad_norm_gen.detach().cpu().numpy()
                ]
                if self.iterCount % int(self.epochCount*len(self.dataLoader)/10) == 0 or self.iterCount == self.epochCount*len(self.dataLoader) - 1:
                    print(f'training: {int(self.iterCount/(self.epochCount*len(self.dataLoader))*100)} %')
                    if isinstance(self.testLabel, int):
                        with no_grad():
                            xFakeTest = self.Gen(self.noiseFixed, self.labelsFixed)
                            yFakeTest = self.Dis(xFakeTest, self.labelsFixed)
                            plt.figure(figsize = (4, 3), facecolor = 'w')
                            plt.plot(xFakeTest.detach().cpu().numpy().T)
                            plt.title(f'labels: {self.labelsFixed.cpu().numpy()}\ndiscriminator: {yFakeTest.detach().cpu().numpy().reshape(-1).round(4)}')
                            plt.show();
                self.iterCount += 1

    def generate_sample(self):
        synthSamples_list = []
        for item in self.dataLoader.dataset.tensors[1]:
            noise = randn(1, self.dimLatent, device = self.device)
            label_ = full(size = (1,), fill_value = item, device = self.device, dtype = torch.int32)
            sampleGen = self.Gen(noise, label_).detach().cpu().numpy()
            synthSamples_list.append(sampleGen)
        synthSamples = np.vstack(synthSamples_list)
        return synthSamples