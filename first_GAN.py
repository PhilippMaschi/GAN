import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from functools import wraps
import time

import math
import matplotlib.pyplot as plt


def performance_counter(func):
    @wraps(func)
    def wrapper(*args):
        t_start = time.perf_counter()
        result = func(*args)
        t_end = time.perf_counter()
        exe_time = round(t_end - t_start, 3)
        print(f"Timer: {func.__name__} - {exe_time}s.")
        return result

    return wrapper

device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


@performance_counter
def main():
    # set a seed for the random generator so the results will be replicable
    torch.manual_seed(111)

    # prepare training data_
    train_data_length = 1024
    # my train data will be a sine curve over 8760 hours
    train_data = torch.zeros((train_data_length, 2))
    # create sine curve with x and y as the two coordinates
    train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)
    train_data[:, 1] = torch.sin(train_data[:, 0])
    train_labels = torch.zeros(train_data_length)
    # pytorch data loader expects a tuples with the training data and the respective label
    train_set = [(train_data[i], train_labels[i]) for i in range(train_data_length)]


    class TrainDataset(Dataset):
        def __init__(self):
            self.train_set = [(train_data[i], train_labels[i]) for i in range(train_data_length)]

        def __len__(self):
            return len(self.train_set)

        def __getitem__(self, index):
            return self.train_set[index]



    plt.plot(train_data[:, 0], train_data[:, 1], ".")
    plt.show()

    # create pytorch data loader
    batch_size = 32  # ! Must be a manyfold by the train data length
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


    # Build Discriminator:
    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                # first layer (input layer)
                nn.Linear(2, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                # second layer
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                # third layer
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                # last layer (output)
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.model(x)


    # Generator
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

        def forward(self, x):
            return self.model(x)


    discriminator = Discriminator()
    generator = Generator()

    # train the models:
    lr = 0.001
    num_epochs = 300
    loss_function = nn.BCELoss()  # binary cross entropy loss function

    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    # training loop
    for epoch in range(num_epochs):
        for n, (real_samples, _) in enumerate(train_loader):
            # data for training the discriminator
            real_samples_labels = torch.ones((batch_size, 1))
            latent_space_samples = torch.randn((batch_size, 2))
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((batch_size, 1))

            # combine real and "fake" samples:
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

            # Train the Discriminator
            discriminator.zero_grad()  # clear the gradients after each training step
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()  # calculate gradients
            optimizer_discriminator.step()  # backpropagation (update weights)

            # data for training the generator
            latent_space_samples = torch.randn((batch_size, 2))  # the generator gets random data as input

            # Train the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()

            # show loss
            if epoch % 10 == 0 and n == batch_size - 1:
                print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                print(f"Epoch: {epoch} Loss G.: {loss_generator}")



    # check how well the model works for a few random input numbers
    latent_space_samples = torch.randn(100, 2)
    generated_samples = generator(latent_space_samples)

    generated_samples = generated_samples.detach()
    plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")


if __name__ == "__main__":
    main()

