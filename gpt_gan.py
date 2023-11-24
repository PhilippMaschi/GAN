import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Load the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Hour'] = data['Date'].dt.hour
    data.drop('Date', axis=1, inplace=True)
    return data

# Normalize the data
def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler, scaler.fit_transform(data)

# Define the generator and discriminator architectures
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Training function
def train_gan(data, generator, discriminator, epochs=1000, batch_size=64, latent_dim=100):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(epochs):
        for i in range(0, len(data), batch_size):
            # Prepare real and fake data batches
            batch_data = data[i:i+batch_size]
            real_data = batch_data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            current_batch_size = real_data.size(0)
            z = torch.randn(current_batch_size, latent_dim).to(real_data.device)
            fake_data = generator(z)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(real_data), torch.ones(current_batch_size, 1).to(real_data.device))
            fake_loss = criterion(discriminator(fake_data.detach()), torch.zeros(current_batch_size, 1).to(real_data.device))
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            g_loss = criterion(discriminator(fake_data), torch.ones(current_batch_size, 1).to(real_data.device))
            g_loss.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{epochs}]  Loss D: {d_loss.item()}, loss G: {g_loss.item()}")

# Main function
def main():
    file_path = 'profiles_test.csv'  # Replace with your file path
    data = load_data(file_path)
    scaler, data_normalized = normalize_data(data)

    input_dim = data_normalized.shape[1]
    latent_dim = 100
    generator = Generator(latent_dim, input_dim)
    discriminator = Discriminator(input_dim)

    data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
    train_gan(data_tensor, generator, discriminator)

    # Save the trained generator model
    torch.save(generator.state_dict(), 'trained_generator.pth')

    # Generate synthetic data
    z = torch.randn(1000, latent_dim)  # Generate 1000 synthetic samples
    synthetic_data = generator(z).detach().numpy()

    # Post-processing to convert synthetic data back to original scale and format
    synthetic_data_rescaled = scaler.inverse_transform(synthetic_data)
    synthetic_data_df = pd.DataFrame(synthetic_data_rescaled, columns=data.columns)

    # Save synthetic data to a CSV file
    synthetic_data_df.to_csv('synthetic_profiles.csv', index=False)

if __name__ == "__main__":
    main()

