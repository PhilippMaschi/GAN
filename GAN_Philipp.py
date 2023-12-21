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
import pytorch_lightning as pl
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
            # # 3rd layer
            # nn.Linear(in_features=self.targetCount * 40, out_features=256),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(inplace=True),
            # # 4th layer
            # nn.Linear(in_features=256, out_features=128),
            # nn.BatchNorm1d(128),
            # nn.LeakyReLU(inplace=True),
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
    # todo LOSS correction, shallow network (128 to 64 after should be able to aprox mean ), if that works add another
    #  layer or make layer deeper, remove dropout or batchnorm (batchnorm might be better), normalize over all profiles,
    #  try VAE,https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f
    #
    def forward(self, data):
        return self.model(data)


class GAN(pl.LightningModule):
    def __init__(self, name, device, batchSize, target, features, dimNoise, featureCount, lr, maxNorm, epochCount,
                 n_transformed_features: int, n_number_features: int, cluster_label: int, cluster_algorithm: str,
                 n_profiles_trained_on: int, LossFct: str):
        super().__init__()
        self.name = name
        # self.device = device
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

        # Initialize generator, input is noise + labels (dimLatent) and output is 24 (target shape)
        self.Gen = Generator(self.dimNoise, self.featureCount, self.target.shape[1])
        # self.Gen.to(self.device)

        # Initialize discriminator
        self.Dis = Discriminator(self.target.shape[1])  # discriminator gets vector with 24 values
        # self.Dis.to(self.device)

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

    def configure_optimizers(self):
        opt_gen = optim.Adam(self.Gen.parameters(), lr=self.lr)
        # opt_dis = optim.Adam(self.Dis.parameters(), lr=self.lr)
        return opt_gen #[opt_gen, opt_dis], []

    def train_discriminator(self, real_data, labels):
        # Train discriminator with real data
        self.Dis.zero_grad()  # set the gradients to zero for every mini-batch
        # yReal: length as target train discriminator with real data, row vector: number of days x profiles
        yReal = self.Dis(real_data)
        labelReal = full(size=(labels.size(0), 1),
                         fill_value=1,
                         # device=self.device,
                         dtype=torch.float32)  # Column vector a tensor containing only ones

        lossDisReal = self.dis_loss_fct(yReal, labelReal)  # calculate the loss of Dis : Single number

        self.log("train_loss_disc_real", lossDisReal)

        # Train with fake data
        noise = torch.randn(real_data.shape[0], self.dimNoise,)# device=self.device)
        # Generate fake labels as in your original code
        first_columns = torch.rand(labels.shape[0], self.n_transformed_features,)# device=self.device) * 2 - 1
        second_columns = torch.randint(0, 2, (labels.shape[0], self.n_number_features), dtype=torch.float32, )# device=self.device,

        fake_labels = torch.cat((first_columns, second_columns), dim=1)

        fake_data = self.Gen(noise, fake_labels)
        fake_labels = torch.full((real_data.size(0), 1), 0.,)# device=self.device)
        fake_output = self.Dis(fake_data.detach())
        fake_loss = self.dis_loss_fct(fake_output, fake_labels)
        self.log('train_loss_disc_fake', fake_loss)

        grad_norm_dis = torch.nn.utils.clip_grad_norm_(self.Dis.parameters(), max_norm=self.maxNorm)
        self.log("norm_grad_dic", grad_norm_dis)
        # Combine losses
        discriminator_loss = lossDisReal + fake_loss
        self.log('train_loss_disc', discriminator_loss)
        return discriminator_loss

    def train_generator(self, labels):
        noise = torch.randn(labels.shape[0], self.dimNoise,)# device=self.device)
        # Generate fake labels as in your original code
        first_columns = torch.rand(labels.shape[0], self.n_transformed_features,) * 2 - 1# device=self.device)
        second_columns = torch.randint(0, 2, (labels.shape[0], self.n_number_features), dtype=torch.float32)#, device=self.device)
        fake_labels = torch.cat((first_columns, second_columns), dim=1)

        fake_data = self.Gen(noise, fake_labels)
        real_labels = torch.full((labels.size(0), 1), 1.,)# device=self.device)
        output = self.Dis(fake_data)
        generator_loss = self.criterion(output, real_labels)
        self.log('train_loss_gen', generator_loss)
        grad_norm_gen = torch.nn.utils.clip_grad_norm_(self.Gen.parameters(), max_norm=self.maxNorm)
        self.log("grad_norm_gen", grad_norm_gen)
        return generator_loss

    def training_step(self, batch, batchidx, optimizer_idx):
        real_data, labels = batch

        if optimizer_idx == 0:
            loss = self.train_discriminator(real_data, labels)
            return {'loss': loss}

        # Training Generator
        if optimizer_idx == 1:
            loss = self.train_generator(labels)
            return {'loss': loss}


def checkpoint_callback(folder_name):
    return pl.callbacks.ModelCheckpoint(
        dirpath=folder_name,  # Replace with your path
        filename='{epoch}-{step}',  # Customizable
        every_n_epochs=500,
        save_top_k=-1,  # -1 indicates all checkpoints are saved
        save_weights_only=True  # Set to False if you want to save the entire model
    )


def train_gan(**kwargs):
    model = GAN(**kwargs)
    model_checkpoint = checkpoint_callback(folder_name=model.folder_name)
    trainer = pl.Trainer(
        callbacks=[model_checkpoint],
        max_epochs=kwargs["epochCount"],
        accelerator="auto",
        devices="auto",
        strategy="auto"
    )
    trainer.fit(model)
    return model


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
