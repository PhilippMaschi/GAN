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
    def __init__(self,
                 target,
                 # features,
                 ):
        # self.features = features  # used to condition the GAN
        self.target = target  # data that should be modelled

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.target[idx]  #, self.features[idx]


class Generator(nn.Module):
    def __init__(self,
                 noise_dim,
                 # feature_dim,
                 target_shape):
        """
        Args:
            noise_dim: is the dimension of the noise vector (which includes the features that are added)
            targetCount: is the output dimension, (24h) in this case
        """
        super(Generator, self).__init__()
        self.target_shape = target_shape
        target_size = torch.prod(torch.tensor(target_shape))
        self.initial_channels = 16
        self.initial_size = 6
        linear_output_size = self.initial_channels * self.initial_size * self.initial_size
        self.linear = nn.Sequential(
            # 1st layer
            nn.Linear(in_features=noise_dim, out_features=linear_output_size),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(self.initial_channels, self.initial_channels // 2, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(self.initial_channels // 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.initial_channels // 2, self.initial_channels // 4, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(self.initial_channels // 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.initial_channels // 4, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, noise):
        # z is the input noise vector
        x = self.linear(noise)
        z = x.view(-1, self.initial_channels, self.initial_size, self.initial_size)
        output = self.conv(z)

        return output.view(-1, *self.target_shape)


class Discriminator(nn.Module):
    def __init__(self, target_shape):
        super(Discriminator, self).__init__()

        target_size = torch.prod(torch.tensor(target_shape))
        # Assuming target_shape is (days, hours)
        batchsize, days, hours = target_shape[0], target_shape[1], target_shape[2]
        kernel_size = 3  # The size of the filter. In 1D convolution, it's the number of time steps the filter covers.
        # For example, a kernel_size of 3 means each filter looks at 3 consecutive time steps in each convolution
        # operation.
        # The stride of the convolution. The stride is the number of time steps the filter moves after each operation.
        # A stride of 1 means the filter moves one step at a time. A larger stride results in downsampling of the input.

        self.model = nn.Sequential(
            # todo try Conv1D, Conv2D
            # n_profiles, (51, 7, 24),  2D or 1D (1D lernt Tage und 2D lernt wochen mit, bezieht sich auf die letzen dimensionen)
            # nn.Conv2d(in_channels=24, out_channels=64, kernel_size=24),  # batch, 365, 1, 64
            # nn.Flatten(),  # batch, 365*16
            # The output from a nn.Conv1d layer is a three-dimensional tensor with shape
            # (batch_size, out_channels, conv_output_length), The number of channels in the input signal. For instance,
            # in time-series data, if you are only looking at one feature (like temperature over time), in_channels
            # would be 1. If you're analyzing multiple features at each time step (like temperature, humidity, and
            # pressure), in_channels would be equal to the number of features. I could use the cluster labels as
            # in_channels!
            # nn.Conv1d(in_channels=1, out_channels=16, kernel_size=kernel_size),
            # nn.LeakyReLU(inplace=True),
            # nn.Flatten(),
            # 1st layer
            nn.Linear(in_features=target_size, out_features=8),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),

            # layer
            nn.Linear(in_features=8, out_features=1),
            nn.Sigmoid()
        )
    # todo LOSS correction, shallow network (128 to 64 after should be able to aprox mean ), if that works add another
    #  layer or make layer deeper, remove dropout or batchnorm (batchnorm might be better), normalize over all profiles,
    #  try VAE,https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f

    def forward(self, data):
        data_flat = data.view(data.size(0), -1)  # Flatten the data
        return self.model(data_flat)


class GAN:
    def __init__(self,
                 name,
                 device,
                 batchSize,
                 target,
                 # features,
                 dimNoise,
                 # featureCount,
                 lr_dis,
                 lr_gen,
                 maxNorm,
                 epochCount,
                 # n_transformed_features: int,
                 # n_number_features: int,
                 cluster_label: int,
                 cluster_algorithm: str,
                 n_profiles_trained_on: int,
                 LossFct: str
                 ):
        super().__init__()
        self.name = name
        self.device = device
        self.batchSize = batchSize
        self.target = target
        # self.features = features
        self.dimNoise = dimNoise
        # self.dimLatent = dimNoise + featureCount  # dimension of noise vector (features are added to noise vector)
        # self.featureCount = featureCount
        self.maxNorm = maxNorm
        self.epochCount = epochCount
        self.lossFct = LossFct
        self.lr_gen = lr_gen
        self.lr_dis = lr_dis

        self.folder_name = f"models/{self.name}_" \
                           f"Clustered={cluster_algorithm}_" \
                           f"ClusterLabel={cluster_label}_" \
                           f"NProfilesTrainedOn={n_profiles_trained_on}_" \
                           f"BatchSize={self.batchSize}_" \
                           f"NoiseDim={self.dimNoise}_" \
                           f"Loss={self.lossFct}_" \
                           f"DisLR={lr_dis}_" \
                           f"GenLR={lr_gen}"

        Path(self.folder_name).mkdir(parents=True, exist_ok=True)
        # if there is files in this folder, delete them
        # for file in Path(self.folder_name).iterdir():
        #     file.unlink()
        # self.n_transformed_features = n_transformed_features
        # self.n_number_features = n_number_features

        target_tensor = torch.Tensor(target)
        # features_tensor = torch.Tensor(self.features)
        self.dataset = MyDataset(target_tensor)  # , features_tensor)
        self.dataLoader = DataLoader(self.dataset, batch_size=self.batchSize, shuffle=True)  # True)

        # Initialize generator, input is noise + labels (dimLatent) and output is 24 (target shape)
        self.Gen = Generator(self.dimNoise, self.target.shape[-2:])
        self.Gen.to(self.device)

        # Initialize discriminator
        self.Dis = Discriminator(self.target.shape)  # discriminator gets vector with 24 values
        self.Dis.to(self.device)

        # Initialize optimizers
        self.optimGen = optim.Adam(params=self.Gen.parameters(), lr=self.lr_gen)
        self.optimDis = optim.Adam(params=self.Dis.parameters(), lr=self.lr_dis)

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

    # def __labels__(self):
    #     return self.features

    def save_model_state(self, checkpoint_path, epoch):
        torch.save({
            "epoch": epoch,
            # "scaler": self.scaler,
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
            "lr_gen": self.lr_gen,
            "lr_dis": self.lr_dis,
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
            for batchIdx, data in enumerate(self.dataLoader):  # target = actual (real) label
                # rows: days x profiles (as provoded by dataLoader => length Batchsize)), columns hours per day
                target_to = data.to(device=self.device, dtype=torch.float32)

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
                # a tensor containing only zeros
                labelFake = full(size=(target_to.size(0), 1),
                                 fill_value=0,
                                 device=self.device,
                                 dtype=torch.float32)
                # create fake data from noise + random labels with generator
                xFake = self.Gen(noise)
                yFake = self.Dis(xFake.detach())  # let the discriminator label the fake data
                lossDisFake = self.dis_loss_fct(yFake, labelFake)
                lossDisFake.backward()

                # lossDis = (lossDisReal + lossDisFake)  # compute the total discriminator loss
                # gradient clipping (large max_norm to avoid actual clipping)
                grad_norm_dis = torch.nn.utils.clip_grad_norm_(self.Dis.parameters(), max_norm=self.maxNorm)
                self.optimDis.step()  # update the discriminator

                # Train generator (now that we fed the discriminator with fake data)
                iterations = 4
                for i in range(iterations):
                    self.Gen.zero_grad()
                    # let the discriminator label the fake data (now that the discriminator is updated)
                    xFake_2 = self.Gen(noise)
                    yFake_2 = self.Dis(xFake_2)
                    # calculate the generator loss (small if the discriminator thinks that `yFake_2 == labelReal`)
                    lossGen = self.criterion(yFake_2, labelReal)
                    if i == iterations-1:
                        lossGen.backward()
                    else:
                        lossGen.backward(retain_graph=True)
                    grad_norm_gen = torch.nn.utils.clip_grad_norm_(self.Gen.parameters(), max_norm=self.maxNorm)
                    self.optimGen.step()

                # Log after every 30 steps
                # if batchIdx % 30 == 0:
                #     run[npt_logger.base_namespace]["batch/lossDisReal"].append(lossDisReal.item())
                #     run[npt_logger.base_namespace]["batch/lossDisFake"].append(lossDisReal.item())
                #     run[npt_logger.base_namespace]["batch/lossGen"].append(lossGen.item())
                #     run[npt_logger.base_namespace]["batch/grad_norm_gen"].append(grad_norm_gen.item())
                #     run[npt_logger.base_namespace]["batch/grad_norm_dis"].append(grad_norm_dis.item())

                # save the model state every 500 epochs:
                if (epoch + 1) % 500 == 0:
                    self.save_model_state(f"{self.folder_name}/epoch={epoch + 1}.pt", epoch)

            # npt_logger.log_checkpoint()
            # Append the losses and gradient norms to the lists
            losses_dis_real.append(lossDisReal.detach().cpu().numpy())
            losses_dis_fake.append(lossDisFake.detach().cpu().numpy())
            losses_gen.append(lossGen.detach().cpu().numpy())
            grad_norms_dis.append(grad_norm_dis.detach().cpu().numpy())
            grad_norms_gen.append(grad_norm_gen.detach().cpu().numpy())


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
        del self.target
        del self.dataset
        del self.dataLoader

# def checkpoint_callback(folder_name):
#     return pl.callbacks.ModelCheckpoint(
#         dirpath=folder_name,  # Replace with your path
#         filename='{epoch}-{step}',  # Customizable
#         every_n_epochs=500,
#         save_top_k=-1,  # -1 indicates all checkpoints are saved
#         save_weights_only=True  # Set to False if you want to save the entire model
#     )


# def train_gan(**kwargs):
#     model = GAN(**kwargs)
#     model_checkpoint = checkpoint_callback(folder_name=model.folder_name)
#     trainer = pl.Trainer(
#         callbacks=[model_checkpoint],
#         max_epochs=kwargs["epochCount"],
#         accelerator="auto",
#         devices="auto",
#         strategy="auto"
#     )
#     trainer.fit(model)
#     return model


def generate_data_from_saved_model(
        model_path,
        noise_dim: int,
        targetShape,
        min_max,
        normalized: bool = False,
):
    """

    Args:
        model_path:
        noise_dim:
        targetShape: (number of profiles, number of days, 24)
        original_features:
        normalized: if normalized the profiles are going to be between -1 and 1
        device:

    Returns:

    """
    # Initialize the generator
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    generator = Generator(noise_dim, target_shape=targetShape[-2:])
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.to(device)
    generator.eval()
    # Generate the data
    with torch.no_grad():
        noise = randn(targetShape[0], noise_dim, device=device, dtype=torch.float32)
        generated_samples = generator(noise).detach().cpu().numpy()
        if not normalized:
            min_val, max_val = min_max[0], min_max[1]
            # Apply min-max scaling inverted
            scaled_samples = generated_samples * (max_val - min_val) + min_val
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
