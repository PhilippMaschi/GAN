import torch
from torch import nn, cat, optim, full, randn, no_grad
import pandas as pd
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from time import perf_counter


# import sys


class Generator(nn.Module):
    def __init__(self, dimLatent, featureCount, classCount, dimEmbedding):
        super(Generator, self).__init__()
        self.dimLatent = dimLatent  # Spalten von noise vector
        self.featureCount = featureCount  # hours per day
        self.classCount = classCount
        self.dimEmbedding = dimEmbedding  # dimension of the embedding tensor
        self.labelEmbedding = nn.Embedding(num_embeddings=self.classCount, embedding_dim=self.dimEmbedding)
        self.model = nn.Sequential(
            # 1st layer
            nn.Linear(in_features=self.dimLatent + self.dimEmbedding, out_features=64),
            nn.LeakyReLU(inplace=True),
            # 2nd layer
            nn.Dropout(0.2),
            # 3rd layer
            nn.Linear(in_features=64, out_features=128),
            nn.LeakyReLU(inplace=True),
            # 4th layer
            nn.Linear(in_features=128, out_features=128),
            nn.LeakyReLU(inplace=True),
            # 5th layer
            nn.Linear(in_features=128, out_features=128),
            nn.LeakyReLU(inplace=True),
            # 6th layer
            nn.Dropout(0.1),
            # 7th layer
            nn.Linear(in_features=128, out_features=featureCount),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        labels_ = self.labelEmbedding(labels)
        # apply model to concatenated tensor (fixed label tensor + noise tensor) noise is added to columns: Rows stay the same
        x = self.model(cat((labels_, noise), -1))
        return x


class Discriminator(nn.Module):
    def __init__(self, featureCount, classCount, dimEmbedding):
        super(Discriminator, self).__init__()
        self.featureCount = featureCount
        self.classCount = classCount
        self.dimEmbedding = dimEmbedding
        self.labelEmbedding = nn.Embedding(num_embeddings=self.classCount, embedding_dim=dimEmbedding)
        self.model = nn.Sequential(
            # 1st layer
            nn.Linear(in_features=self.featureCount + self.dimEmbedding, out_features=128),
            nn.LeakyReLU(inplace=True),
            # 2nd layer
            nn.Dropout(0.1),
            # 3rd layer
            nn.Linear(in_features=128, out_features=128),
            nn.LeakyReLU(inplace=True),
            # 4th layer
            nn.Linear(in_features=128, out_features=128),
            nn.LeakyReLU(inplace=True),
            # 5th layer
            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(inplace=True),
            # 6th layer
            nn.Dropout(0.1),
            # 7th layer
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, data, labels):
        labels_ = self.labelEmbedding(labels)
        bool_ = self.model(cat((data, labels_), -1))
        return bool_


class GAN(object):
    def __init__(self, name, device, batchSize, samples, labels, dimLatent, featureCount, classCount, dimEmbedding, lr,
                 maxNorm, epochCount, testLabel=None, exampleCount=3):
        self.name = name
        self.device = device
        self.batchSize = batchSize
        self.samples = samples
        self.labels = labels
        self.dimLatent = dimLatent
        self.featureCount = featureCount
        self.classCount = classCount
        self.dimEmbedding = dimEmbedding
        self.lr = lr
        self.maxNorm = maxNorm
        self.epochCount = epochCount
        self.testLabel = testLabel
        self.exampleCount = exampleCount
        self.file_name = f"{self.name}_batchSize={self.batchSize}_samples={self.samples}_labels={self.labels}_" \
                         f"dimLatent={self.dimLatent}_featureCount={self.featureCount}_classCount={self.classCount}_" \
                         f"dimEmbedding={self.dimEmbedding}"

        # Scale data and create dataLoader
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.samplesScaled = self.scaler.fit_transform(samples.T).T
        samples_ = torch.Tensor(self.samplesScaled)
        labels_ = torch.Tensor(self.labels)
        self.dataset = TensorDataset(samples_, labels_)
        self.dataLoader = DataLoader(self.dataset, batch_size=self.batchSize, shuffle=True)  # True)

        # Initialize generator, classCount all profiles x days, not batched
        self.Gen = Generator(dimLatent, featureCount, classCount, self.dimEmbedding)
        self.Gen.to(self.device)

        # Initialize discriminator
        self.Dis = Discriminator(featureCount, classCount, self.dimEmbedding)
        self.Dis.to(self.device)

        # Initialize optimizers
        self.optimGen = optim.Adam(params=self.Gen.parameters(), lr=self.lr)
        self.optimDis = optim.Adam(params=self.Dis.parameters(), lr=self.lr)

        # Initialize the loss function
        self.criterion = nn.BCELoss()

        self.df_loss = pd.DataFrame(
            columns=[
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
            self.noiseFixed = randn(self.exampleCount, dimLatent, device=device)
            self.labelsFixed = full(size=(self.exampleCount,), fill_value=self.testLabel, device=self.device,
                                    dtype=torch.int32)

    def __labels__(self):
        return self.labels

    def save_model_state(self, checkpoint_path, epoch):
        torch.save({
            "epoch": epoch,
            'generator_state_dict': self.Gen.state_dict(),
            'discriminator_state_dict': self.Dis.state_dict(),
            'optimizer_gen_state_dict': self.optimGen.state_dict(),
            'optimizer_dis_state_dict': self.optimDis.state_dict(),
            "label": self.labels
        }, checkpoint_path)

    def load_model_state(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.Gen.load_state_dict(checkpoint['generator_state_dict'])
        self.Dis.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimGen.load_state_dict(checkpoint['optimizer_gen_state_dict'])
        self.optimDis.load_state_dict(checkpoint['optimizer_dis_state_dict'])
        print(f"loaded model at epoch: {checkpoint['epoch']}")

    def train(self):
        for epoch in tqdm(range(self.epochCount)):
            for batchIdx, (data, target_) in enumerate(self.dataLoader):  # target = actual (real) label
                # rows: days x profiles (as provoded by dataLoader => length Batchsize)), columns hours per day
                data = data.to(device=self.device, dtype=torch.float32)
                # Index column vector: rows are days x profiles (as provoded by dataLoader => length Batchsize))
                target = target_.to(device=self.device, dtype=torch.int32)

                # Train discriminator with real data
                tstamp_1 = perf_counter()
                self.Dis.zero_grad()  # set the gradients to zero for every mini-batch
                # yReal: length as target train discriminator with real data, row vector: number of days x profiles
                yReal = self.Dis(data, target)
                labelReal = full(size=(data.size(0), 1),
                                 fill_value=1,
                                 device=self.device,
                                 dtype=torch.float32)  # Column vector a tensor containing only ones
                lossDisReal = self.criterion(yReal, labelReal)  # calculate the loss : Single number
                lossDisReal.backward()  # calculate new gradients
                # print(f'Train discriminator with real data: {perf_counter() - tstamp_1}')
                # Train discriminator with fake data
                tstamp_2 = perf_counter()
                # create a tensor filled with random numbers rows: Number of days, column dimLatent
                noise = randn(data.size(0), self.dimLatent, device=self.device)
                randomLabelFake = target_.to(device=self.device,
                                             dtype=torch.int32)  # torch.randint(low = 0, high = self.classCount, size = (data.size(0),), device = self.device)  #random labels needed in addition to the noise
                labelFake = full(size=(data.size(0), 1), fill_value=0, device=self.device,
                                 dtype=torch.float32)  # a tensor containing only zeros
                xFake = self.Gen(noise, randomLabelFake)  # create fake data from noise + random labels with generator
                yFake = self.Dis(xFake.detach(),
                                 randomLabelFake)  # let the discriminator label the fake data (`.detach()` creates a copy of the tensor)
                lossDisFake = self.criterion(yFake, labelFake)
                lossDisFake.backward()

                lossDis = (lossDisReal + lossDisFake)  # compute the total discriminator loss
                grad_norm_dis = torch.nn.utils.clip_grad_norm_(self.Dis.parameters(),
                                                               max_norm=self.maxNorm)  # gradient clipping (large max_norm to avoid actual clipping)
                self.optimDis.step()  # update the discriminator
                # print(f'Train discriminator with fake data: {perf_counter() - tstamp_2}')

                # Train generator (now that we fed the discriminator with fake data)
                tstamp_3 = perf_counter()
                self.Gen.zero_grad()
                yFake_2 = self.Dis(xFake,
                                   randomLabelFake)  # let the discriminator label the fake data (now that the discriminator is updated)
                lossGen = self.criterion(yFake_2,
                                         labelReal)  # calculate the generator loss (small if the discriminator thinks that `yFake_2 == labelReal`)
                lossGen.backward()
                grad_norm_gen = torch.nn.utils.clip_grad_norm_(self.Gen.parameters(), max_norm=self.maxNorm)
                self.optimGen.step()
                # print(f'Train generator (now that we fed the discriminator with fake data): {perf_counter() - tstamp_3}')

                # save the model state every 500 epochs:
                if (epoch + 1) % 100 == 0:
                    self.save_model_state(f"models/{self.file_name}_epoch={epoch + 1}.pt", epoch)

                # Log the progress
                tstamp_4 = perf_counter()
                # self.df_loss.loc[len(self.df_loss)] = [
                #     epoch,
                #     batchIdx,
                #     lossDisReal.detach().cpu().numpy(),
                #     lossDisFake.detach().cpu().numpy(),
                #     lossDis.detach().cpu().numpy(),
                #     lossGen.detach().cpu().numpy(),
                #     grad_norm_dis.detach().cpu().numpy(),
                #     grad_norm_gen.detach().cpu().numpy()
                # ]
                # if self.iterCount % max(1, int(self.epochCount * len(
                #         self.dataLoader) / 10)) == 0 or self.iterCount == self.epochCount * len(self.dataLoader) - 1:
                #     # print(f'training: {int(self.iterCount/(self.epochCount*len(self.dataLoader))*100)} %')
                #     if isinstance(self.testLabel, int):
                #         with no_grad():
                #             xFakeTest = self.Gen(self.noiseFixed, self.labelsFixed)
                #             yFakeTest = self.Dis(xFakeTest, self.labelsFixed)
                #             plt.figure(figsize=(4, 3), facecolor='w')
                #             plt.plot(xFakeTest.detach().cpu().numpy().T)
                #             plt.title(
                #                 f'labels: {self.labelsFixed.cpu().numpy()}\ndiscriminator: {yFakeTest.detach().cpu().numpy().reshape(-1).round(4)}')
                #             plt.show();
                # self.iterCount += 1

        del self.samples
        del self.samplesScaled
        del self.dataset
        del self.dataLoader

    def generate_sample(self, labels: np.array):
        with torch.no_grad():
            noise = randn(len(self.labels), self.dimLatent, device=self.device)
        return self.Gen(noise, self.labels.to(device=self.device, dtype=torch.int32)).detach().to_dense().cpu().numpy()

    def generate_scaled_sample(self, labels: np.array):
        synthSamples = self.generate_sample(self.labels)
        scaled_gen_sample = self.scaler.inverse_transform(synthSamples.T).T
        return scaled_gen_sample


def generate_data_from_saved_model(
        model_path,
        dim_latent,
        featureCount,
        class_count,
        dim_embedding,
        number_of_profiles: int,
        device='cpu'
):
    # Initialize the generator
    generator = Generator(dim_latent, featureCount, class_count, dim_embedding)
    generator.load_state_dict(torch.load(model_path)['generator_state_dict'])
    generator.to(device)
    generator.eval()
    # Generate the data
    with torch.no_grad():
        training_labels = np.tile(np.array(range(class_count)), number_of_profiles)
        noise = torch.randn(len(training_labels), dim_latent, device=device)
        labels = torch.tensor(training_labels, device=device)  # Example: Random labels
        generated_samples = generator(noise, labels).detach().cpu().numpy()
    return generated_samples

if __name__ == "__main__":
    generate_data_from_saved_model(
        model_path=f"models/model_test_philipp_GAN_epoch_10.pt",
        dim_latent=10,
        featureCount=24,
        class_count=395,
        dim_embedding=100,
        number_of_profiles=10,
        device='cpu'
    )
