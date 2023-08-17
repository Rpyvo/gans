#Bu kod reshy tarafından onrir için yazılmıştır
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

lr = 0.001
z_dim = 64
data_dim = 28 * 28

generator = Generator(z_dim, data_dim)
discriminator = Discriminator(data_dim)
generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

batch_size = 64
train_data = torch.randn(batch_size, z_dim)

num_epochs = 100
for epoch in range(num_epochs):
    for _ in range(len(train_data)):
        # Discriminator'ı güncelle
        real_data = torch.randn(batch_size, data_dim)  # Gerçek veri yerine rastgele veri kullanılıyor
        discriminator_loss_real = nn.BCELoss()(discriminator(real_data), torch.ones(batch_size, 1))
        
        fake_data = generator(train_data)
        discriminator_loss_fake = nn.BCELoss()(discriminator(fake_data), torch.zeros(batch_size, 1))
        
        discriminator_loss = discriminator_loss_real + discriminator_loss_fake
        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()
        
        fake_data = generator(train_data)
        generator_loss = nn.BCELoss()(discriminator(fake_data), torch.ones(batch_size, 1))
        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] - Generator Loss: {generator_loss.item():.4f}, Discriminator Loss: {discriminator_loss.item():.4f}")

generated_data = generator(torch.randn(16, z_dim))
generated_data = generated_data.view(-1, 28, 28).detach().numpy()

plt.figure(figsize=(4, 4))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated_data[i], cmap='gray')
    plt.axis('off')
plt.show()
