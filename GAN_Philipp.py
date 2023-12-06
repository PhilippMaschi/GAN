import torch
from torch import nn, cat, optim, full, randn, no_grad
import pandas as pd
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from time import perf_counter


# import pytorch.lightning


# import sys


class MyDataset(Dataset):
    def __init__(self, target, features, ):
        self.features = features  # used to condition the GAN
        self.target = target  # data that should be modelled

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.target[idx], self.features[idx]


class Generator(nn.Module):
    def __init__(self, noise_dim, feature_dim, targetCount):
        """
        Args:
            noise_dim: is the dimension of the noise vector (which includes the features that are added)
            targetCount: is the output dimension, (24h) in this case
        """
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # 1st layer
            nn.Linear(in_features=noise_dim+feature_dim, out_features=noise_dim * 8),
            nn.LeakyReLU(inplace=True),
            # 2nd layer
            nn.Dropout(0.2),
            # 3rd layer
            nn.Linear(in_features=noise_dim * 8, out_features=128),
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
            nn.Linear(in_features=128, out_features=targetCount),
            nn.Tanh()
        )

    def forward(self, noise, features):
        return self.model(cat((noise, features), -1))


class Discriminator(nn.Module):
    def __init__(self, targetCount):
        super(Discriminator, self).__init__()
        self.targetCount = targetCount
        self.model = nn.Sequential(
            # 1st layer
            nn.Linear(in_features=self.targetCount, out_features=self.targetCount * 4),
            nn.LeakyReLU(inplace=True),
            # 2nd layer
            nn.Dropout(0.1),
            # 3rd layer
            nn.Linear(in_features=self.targetCount * 4, out_features=128),
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

    def forward(self, data):
        return self.model(data)


class GAN(object):
    def __init__(self,
                 name,
                 device,
                 batchSize,
                 target,
                 features,
                 dimNoise,
                 featureCount,
                 lr,
                 maxNorm,
                 epochCount,
                 n_transformed_features: int,
                 n_number_features: int,
                 testLabel=None,
                 exampleCount=3,
                 ):
        self.name = name
        self.device = device
        self.batchSize = batchSize
        self.target = target
        self.features = features
        self.dimNoise = dimNoise
        # self.dimLatent = dimNoise + featureCount  # dimension of noise vector (features are added to noise vector)
        self.featureCount = featureCount
        self.lr = lr
        self.maxNorm = maxNorm
        self.epochCount = epochCount
        self.testLabel = testLabel
        self.exampleCount = exampleCount
        self.file_name = f"{self.name}_" \
                         f"batchSize={self.batchSize}_" \
                         f"featureCount={self.featureCount}"
        self.n_transformed_features = n_transformed_features
        self.n_number_features = n_number_features

        # Scale data and create dataLoader
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.samplesScaled = self.scaler.fit_transform(target.T).T
        target_tensor = torch.Tensor(self.samplesScaled)
        features_tensor = torch.Tensor(self.features)
        self.dataset = MyDataset(target_tensor, features_tensor)
        self.dataLoader = DataLoader(self.dataset, batch_size=self.batchSize, shuffle=True)  # True)

        # Initialize generator
        self.Gen = Generator(self.dimNoise, self.featureCount, self.target.shape[1])  # input is noise + labels (dimLatent) and output is 24 (target shape)
        self.Gen.to(self.device)

        # Initialize discriminator
        self.Dis = Discriminator(self.target.shape[1])  # discriminator gets vector with 24 values
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
            self.noiseFixed = randn(self.exampleCount, self.dimLatent, device=device)
            self.labelsFixed = full(size=(self.exampleCount,), fill_value=self.testLabel, device=self.device,
                                    dtype=torch.int32)

    def __labels__(self):
        return self.features

    def save_model_state(self, checkpoint_path, epoch):
        torch.save({
            "epoch": epoch,
            'generator_state_dict': self.Gen.state_dict(),
            'discriminator_state_dict': self.Dis.state_dict(),
            'optimizer_gen_state_dict': self.optimGen.state_dict(),
            'optimizer_dis_state_dict': self.optimDis.state_dict(),
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
            for batchIdx, (data, label) in enumerate(self.dataLoader):  # target = actual (real) label
                # rows: days x profiles (as provoded by dataLoader => length Batchsize)), columns hours per day
                target_to = data.to(device=self.device, dtype=torch.float32)
                # Index column vector: rows are days x profiles (as provoded by dataLoader => length Batchsize))
                feature_to = label.to(device=self.device, dtype=torch.float32)

                # Train discriminator with real data
                tstamp_1 = perf_counter()
                self.Dis.zero_grad()  # set the gradients to zero for every mini-batch
                # yReal: length as target train discriminator with real data, row vector: number of days x profiles
                yReal = self.Dis(target_to)
                labelReal = full(size=(target_to.size(0), 1),
                                 fill_value=1,
                                 device=self.device,
                                 dtype=torch.float32)  # Column vector a tensor containing only ones
                lossDisReal = self.criterion(yReal, labelReal)  # calculate the loss of Dis : Single number
                lossDisReal.backward()  # calculate new gradients
                # print(f'Train discriminator with real data: {perf_counter() - tstamp_1}')
                # Train discriminator with fake data
                tstamp_2 = perf_counter()
                # create a tensor filled with random numbers rows: Number of days, column dimLatent
                noise = randn(target_to.shape[0], self.dimNoise, device=self.device)
                # random labels needed in addition to the noise
                first_columns = torch.rand(feature_to.shape[0], self.n_transformed_features, device=self.device) * 2 - 1
                second_columns = torch.randint(0, 2, (feature_to.shape[0], self.n_number_features), device=self.device, dtype=torch.float32)
                randomLabelFake = torch.cat((first_columns, second_columns), dim=1)
                # a tensor containing only zeros
                labelFake = full(size=(target_to.size(0), 1), fill_value=0, device=self.device, dtype=torch.float32)
                # create fake data from noise + random labels with generator
                xFake = self.Gen(noise, randomLabelFake)
                yFake = self.Dis(xFake.detach())  # let the discriminator label the fake data
                lossDisFake = self.criterion(yFake, labelFake)
                lossDisFake.backward()

                lossDis = (lossDisReal + lossDisFake)  # compute the total discriminator loss
                # gradient clipping (large max_norm to avoid actual clipping)
                grad_norm_dis = torch.nn.utils.clip_grad_norm_(self.Dis.parameters(), max_norm=self.maxNorm)
                self.optimDis.step()  # update the discriminator
                # print(f'Train discriminator with fake data: {perf_counter() - tstamp_2}')

                # Train generator (now that we fed the discriminator with fake data)
                tstamp_3 = perf_counter()
                self.Gen.zero_grad()
                # let the discriminator label the fake data (now that the discriminator is updated)
                yFake_2 = self.Dis(xFake)
                # calculate the generator loss (small if the discriminator thinks that `yFake_2 == labelReal`)
                lossGen = self.criterion(yFake_2, labelReal)
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

        del self.target
        del self.samplesScaled
        del self.dataset
        del self.dataLoader

    def generate_sample(self, labels: np.array):
        with torch.no_grad():
            noise = randn(len(self.features), self.dimLatent, device=self.device)
        return self.Gen(noise,
                        self.features.to(device=self.device, dtype=torch.int32)).detach().to_dense().cpu().numpy()

    def generate_scaled_sample(self, labels: np.array):
        synthSamples = self.generate_sample(self.features)
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
