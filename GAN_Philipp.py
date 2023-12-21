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
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
# from neptune_pytorch import NeptuneLogger
# import neptune
# from neptune.utils import stringify_unsupported

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
            nn.Linear(in_features=noise_dim + feature_dim, out_features=noise_dim * 80),
            nn.BatchNorm1d(noise_dim * 80),
            nn.LeakyReLU(inplace=True),
            # 2nd layer
            nn.Dropout(0.2),
            # 3rd layer
            nn.Linear(in_features=noise_dim * 80, out_features=1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(inplace=True),
            # 4th layer
            nn.Linear(in_features=1000, out_features=500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(inplace=True),
            # 5th layer
            nn.Linear(in_features=500, out_features=128),
            nn.BatchNorm1d(128),
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
            nn.Linear(in_features=self.targetCount, out_features=self.targetCount * 40),
            nn.BatchNorm1d(self.targetCount * 40),
            nn.LeakyReLU(inplace=True),
            # 2nd layer
            nn.Dropout(0.1),
            # 3rd layer
            nn.Linear(in_features=self.targetCount * 40, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            # 4th layer
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            # 5th layer
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
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
                 target,  # train data
                 features,  # labels
                 dimNoise,  # noise vector size
                 featureCount,  # anzahl der features
                 lr,
                 maxNorm,
                 epochCount,
                 n_transformed_features: int,  # anzahl an sin cos transformierten variablen (1 variable wird zu 2 (sin + cos))
                 n_number_features: int,   # anzahl der features (labels) die nicht transformiert werden
                 cluster_label: int,  # label of the cluster
                 cluster_algorithm: str,
                 n_profiles_trained_on: int,
                 LossFct: str
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
        self.lossFct = LossFct

        self.folder_name = f"models/{self.name}_" \
                           f"Clustered={cluster_algorithm}_" \
                           f"ClusterLabel={cluster_label}_" \
                           f"NProfilesTrainedOn={n_profiles_trained_on}_" \
                           f"BatchSize={self.batchSize}_" \
                           f"FeatureCount={self.featureCount}_" \
                           f"NoiseDim={self.dimNoise}_" \
                           f"Loss={self.lossFct}"

        Path(self.folder_name).mkdir(parents=True, exist_ok=True)
        # if there is files in this folder, delete them
        # for file in Path(self.folder_name).iterdir():
        #     file.unlink()
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
        self.Gen = Generator(self.dimNoise, self.featureCount, self.target.shape[
            1])  # input is noise + labels (dimLatent) and output is 24 (target shape)
        self.Gen.to(self.device)

        # Initialize discriminator
        self.Dis = Discriminator(self.target.shape[1])  # discriminator gets vector with 24 values
        self.Dis.to(self.device)

        # Initialize optimizers
        self.optimGen = optim.Adam(params=self.Gen.parameters(), lr=self.lr)
        self.optimDis = optim.Adam(params=self.Dis.parameters(), lr=self.lr)

        # Initialize the loss function
        if self.lossFct == "BCE":
            self.criterion = nn.BCELoss()
        elif self.lossFct == "MSE":
            self.criterion = nn.MSELoss()
        elif self.lossFct == "KLDiv":
            self.criterion = nn.KLDivLoss()
        elif self.lossFct == "MAE":
            self.criterion = nn.L1Loss()
        else:
            assert "loss function not identified"
        self.dis_loss_fct = nn.BCELoss()  # for discriminator always BCE?

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

    def __labels__(self):
        return self.features

    def save_model_state(self, checkpoint_path, epoch):
        torch.save({
            "epoch": epoch,
            "scaler": self.scaler,
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
        print(f"starting training for {self.folder_name}")
        # run = neptune.init_run(
        #     project="philmaschi/GAN",
        #     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ZDE3NGVjNy1kZjdiLTQ1MzMtOGEzNi0yZDhlZjIxZjRjZGIifQ==",
        # )
        # npt_logger = NeptuneLogger(
        #     run=run,
        #     model=self.Gen,
        #     log_model_diagram=True,
        #     log_gradients=True,
        #     log_parameters=True,
        #     log_freq=30,
        # )
        parameters = {
            "lr": self.lr,
            "BatchSize": self.batchSize,
            "NoiseDim": self.dimNoise,
            "model_filename": self.name,
            "device": self.device,
            "epochs": self.epochCount,
            "loss": self.lossFct
        }
        # run[npt_logger.base_namespace]["hyperparams"] = stringify_unsupported(parameters)

        losses_dis_real = []
        losses_dis_fake = []
        losses_gen = []
        grad_norms_dis = []
        grad_norms_gen = []
        for epoch in tqdm(range(self.epochCount)):
            for batchIdx, (data, label) in enumerate(self.dataLoader):  # target = actual (real) label
                # rows: days x profiles (as provoded by dataLoader => length Batchsize)), columns hours per day
                target_to = data.to(device=self.device, dtype=torch.float32)
                # Index column vector: rows are days x profiles (as provoded by dataLoader => length Batchsize))
                feature_to = label.to(device=self.device, dtype=torch.float32)

                # Train discriminator with real data
                self.Dis.zero_grad()  # set the gradients to zero for every mini-batch
                # yReal: length as target train discriminator with real data, row vector: number of days x profiles
                yReal = self.Dis(target_to)
                labelReal = full(size=(target_to.size(0), 1),
                                 fill_value=1,
                                 device=self.device,
                                 dtype=torch.float32)  # Column vector a tensor containing only ones

                lossDisReal = self.dis_loss_fct(yReal, labelReal)  # calculate the loss of Dis : Single number
                lossDisReal.backward()  # calculate new gradients
                # Train discriminator with fake data
                # create a tensor filled with random numbers rows: Number of days, column dimLatent
                noise = randn(target_to.shape[0], self.dimNoise, device=self.device)
                # random labels needed in addition to the noise
                first_columns = torch.rand(feature_to.shape[0], self.n_transformed_features, device=self.device) * 2 - 1
                second_columns = torch.randint(0, 2, (feature_to.shape[0], self.n_number_features), device=self.device,
                                               dtype=torch.float32)
                randomLabelFake = torch.cat((first_columns, second_columns), dim=1)
                # a tensor containing only zeros
                labelFake = full(size=(target_to.size(0), 1), fill_value=0, device=self.device, dtype=torch.float32)
                # create fake data from noise + random labels with generator
                xFake = self.Gen(noise, randomLabelFake)
                yFake = self.Dis(xFake.detach())  # let the discriminator label the fake data
                lossDisFake = self.dis_loss_fct(yFake, labelFake)
                lossDisFake.backward()

                # lossDis = (lossDisReal + lossDisFake)  # compute the total discriminator loss
                # gradient clipping (large max_norm to avoid actual clipping)
                grad_norm_dis = torch.nn.utils.clip_grad_norm_(self.Dis.parameters(), max_norm=self.maxNorm)
                self.optimDis.step()  # update the discriminator

                # Train generator (now that we fed the discriminator with fake data)
                self.Gen.zero_grad()
                # let the discriminator label the fake data (now that the discriminator is updated)
                yFake_2 = self.Dis(xFake)
                # calculate the generator loss (small if the discriminator thinks that `yFake_2 == labelReal`)
                lossGen = self.criterion(yFake_2, labelReal)
                lossGen.backward()
                grad_norm_gen = torch.nn.utils.clip_grad_norm_(self.Gen.parameters(), max_norm=self.maxNorm)
                self.optimGen.step()

                # save the model state every 500 epochs:
                # Log after every 30 steps
                # if batchIdx % 30 == 0:
                #     run[npt_logger.base_namespace]["batch/lossDisReal"].append(lossDisReal.item())
                #     run[npt_logger.base_namespace]["batch/lossDisFake"].append(lossDisReal.item())
                #     run[npt_logger.base_namespace]["batch/lossGen"].append(lossGen.item())
                #     run[npt_logger.base_namespace]["batch/grad_norm_gen"].append(grad_norm_gen.item())
                #     run[npt_logger.base_namespace]["batch/grad_norm_dis"].append(grad_norm_dis.item())

                if (epoch + 1) % 500 == 0:
                    self.save_model_state(f"{self.folder_name}/epoch={epoch + 1}.pt", epoch)

            # npt_logger.log_checkpoint()
            # Append the losses and gradient norms to the lists
            losses_dis_real.append(lossDisReal.detach().cpu().numpy())
            losses_dis_fake.append(lossDisFake.detach().cpu().numpy())
            losses_gen.append(lossGen.detach().cpu().numpy())
            grad_norms_dis.append(grad_norm_dis.detach().cpu().numpy())
            grad_norms_gen.append(grad_norm_gen.detach().cpu().numpy())


        del self.target
        del self.samplesScaled
        del self.dataset
        del self.dataLoader
        # After training
        fig = plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(losses_dis_real, label='Discriminator Loss - Real')
        plt.plot(losses_dis_fake, label='Discriminator Loss - Fake')
        plt.plot(losses_gen, label='Generator Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(grad_norms_dis, label='Discriminator Gradient Norm')
        plt.plot(grad_norms_gen, label='Generator Gradient Norm')
        plt.xlabel('Iterations')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norms During Training')
        plt.legend()

        plt.tight_layout()
        path = Path(__file__).parent / "plots" / f"{Path(self.folder_name).stem}"
        path.mkdir(exist_ok=True, parents=True)
        plt.savefig(Path(__file__).parent / "plots" / f"{Path(self.folder_name).stem}" / "Losses_and_GradientNorm.png")
        plt.close(fig)

        # run.stop()

        del losses_dis_real
        del losses_dis_fake
        del losses_gen
        del grad_norms_dis
        del grad_norms_gen

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
        noise_dim: int,
        featureCount: int,  # number of features that are added to noise vector
        targetCount: int,  # 24
        original_features: np.array,
        normalized: bool = False,
        device='cpu'
):
    """

    Args:
        model_path:
        noise_dim:
        featureCount:
        targetCount:
        original_features:
        normalized: if normalized the profiles are going to be between -1 and 1
        device:

    Returns:

    """
    # Initialize the generator
    generator = Generator(noise_dim, featureCount, targetCount)
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.to(device)
    generator.eval()
    if not normalized:
        scaler = checkpoint["scaler"]
    # Generate the data
    with torch.no_grad():
        noise = torch.randn(len(original_features), noise_dim, device=device, dtype=torch.float32)
        labels = torch.tensor(original_features, device=device, dtype=torch.float32)  # Example: Random labels
        generated_samples = generator(noise, labels).detach().cpu().numpy()
        if not normalized:
            scaled_samples = scaler.inverse_transform(generated_samples.T).T
        else:
            scaled_samples = generated_samples

    return scaled_samples


if __name__ == "__main__":
    generate_data_from_saved_model(
        model_path=f"models/model_test_philipp_GAN_epoch_10.pt",
        noise_dim=10,
        featureCount=3,
        targetCount=24,
        original_features=None,
        device='cpu'
    )
