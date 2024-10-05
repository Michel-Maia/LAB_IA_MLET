import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
import mlflow
import mlflow.pytorch

from src.models import Generator, Discriminator


def save_generated_images(images, epoch, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    utils.save_image(images, os.path.join(output_dir, f'fake_samples_epoch_{epoch}.png'), normalize=True)


def main(config):
    # MLflow experiment setup
    mlflow.set_experiment("DCGAN_Fine_Tuning")
    with mlflow.start_run():
        mlflow.log_params(config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlflow.log_param("device", device.type)
        print(f'Using device: {device}')

        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5 for _ in range(3)], [0.5 for _ in range(3)])
        ])

        # Load dataset
        data_dir = config['data_dir']
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

        # Initialize models
        nz = config['nz']
        ngf = config['ngf']
        ndf = config['ndf']
        nc = config['nc']

        generator = Generator(nz, ngf, nc).to(device)
        discriminator = Discriminator(nc, ndf).to(device)

        # Load pre-trained weights if available
        if config['pretrained']:
            if os.path.exists(config['pretrained_generator']):
                generator.load_state_dict(torch.load(config['pretrained_generator'], map_location=device))
                print("Loaded pre-trained Generator weights.")
                mlflow.log_param("generator_pretrained", config['pretrained_generator'])
            if os.path.exists(config['pretrained_discriminator']):
                discriminator.load_state_dict(torch.load(config['pretrained_discriminator'], map_location=device))
                print("Loaded pre-trained Discriminator weights.")
                mlflow.log_param("discriminator_pretrained", config['pretrained_discriminator'])

        # Loss and optimizers
        criterion = nn.BCELoss()
        optimizerD = optim.Adam(discriminator.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
        optimizerG = optim.Adam(generator.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))

        # Labels
        real_label = 1.
        fake_label = 0.

        # Training Loop
        num_epochs = config['num_epochs']
        output_dir = config['output_dir']

        for epoch in range(num_epochs):
            for i, (data, _) in enumerate(dataloader):
                ############################
                # (1) Update Discriminator
                ###########################
                discriminator.zero_grad()
                # Train with real data
                real_data = data.to(device)
                b_size = real_data.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                output = discriminator(real_data)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                # Train with fake data
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = generator(noise)
                label.fill_(fake_label)
                output = discriminator(fake.detach())
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()

                errD = errD_real + errD_fake
                optimizerD.step()

                ############################
                # (2) Update Generator
                ###########################
                generator.zero_grad()
                label.fill_(real_label)  # Generator tries to fool Discriminator
                output = discriminator(fake)
                errG = criterion(output, label)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

                # Logging
                if i % config['log_interval'] == 0:
                    print(f'[{epoch+1}/{num_epochs}][{i}/{len(dataloader)}] '
                          f'Loss_D: {errD.item():.4f} '
                          f'Loss_G: {errG.item():.4f} '
                          f'D(x): {D_x:.4f} '
                          f'D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
                    mlflow.log_metric("Loss_D", errD.item(), step=epoch * len(dataloader) + i)
                    mlflow.log_metric("Loss_G", errG.item(), step=epoch * len(dataloader) + i)
                    mlflow.log_metric("D_x", D_x, step=epoch * len(dataloader) + i)
                    mlflow.log_metric("D_G_z1", D_G_z1, step=epoch * len(dataloader) + i)
                    mlflow.log_metric("D_G_z2", D_G_z2, step=epoch * len(dataloader) + i)

            # Generate and save images after each epoch
            with torch.no_grad():
                fixed_noise = torch.randn(64, nz, 1, 1, device=device)
                fake_images = generator(fixed_noise).detach().cpu()
            save_generated_images(fake_images, epoch + 1, output_dir)
            mlflow.log_artifact(os.path.join(output_dir, f'fake_samples_epoch_{epoch + 1}.png'))

            # Save model checkpoints
            os.makedirs(config['model_dir'], exist_ok=True)
            generator_path = os.path.join(config['model_dir'], f'generator_epoch_{epoch + 1}.pth')
            discriminator_path = os.path.join(config['model_dir'], f'discriminator_epoch_{epoch + 1}.pth')
            torch.save(generator.state_dict(), generator_path)
            torch.save(discriminator.state_dict(), discriminator_path)
            mlflow.log_artifact(generator_path)
            mlflow.log_artifact(discriminator_path)

        # Log the final Generator model
        mlflow.pytorch.log_model(generator, "generator_model")
        mlflow.pytorch.log_model(discriminator, "discriminator_model")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DCGAN Training with MLflow")
    parser.add_argument('--data_dir', type=str, default='../data/celeba/img_align_celeba', help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='../output', help='Directory to save generated images')
    parser.add_argument('--model_dir', type=str, default='../models', help='Directory to save model checkpoints')
    parser.add_argument('--pretrained', action='store_true', help='Use pre-trained models')
    parser.add_argument('--pretrained_generator', type=str, default='../models/generator.pth', help='Path to pre-trained Generator')
    parser.add_argument('--pretrained_discriminator', type=str, default='../models/discriminator.pth', help='Path to pre-trained Discriminator')
    parser.add_argument('--batch_size', type=int, default=128, help='Input batch size')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    parser.add_argument('--nz', type=int, default=100, help='Size of z latent vector')
    parser.add_argument('--ngf', type=int, default=64, help='Generator feature map size')
    parser.add_argument('--ndf', type=int, default=64, help='Discriminator feature map size')
    parser.add_argument('--nc', type=int, default=3, help='Number of channels in the training images')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--log_interval', type=int, default=100, help='Interval for logging metrics')

    args = parser.parse_args()

    # Convert args to a dictionary
    config = vars(args)

    main(config)
