import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import time


class FontTrainer:
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, device, num_epochs, checkpoint_path):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            start_time = time.time()

            for step, images in enumerate(self.train_loader):
                images = images.to(self.device)
                self.optimizer.zero_grad()

                # Assume src and tgt are obtained from images
                src = images
                tgt = images

                output = self.model(src, tgt)
                loss = self.criterion(output, images)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                # Progress display
                if step % 10 == 0:
                    elapsed_time = time.time() - start_time
                    time_left = (elapsed_time / (step + 1)) * (len(self.train_loader) - step - 1)
                    self._progress(step, running_loss / (step + 1), time_left)

            avg_loss = running_loss / len(self.train_loader)
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss:.4f}')

            self._save_checkpoint(epoch)

            self._valid_iter(epoch)

    def _progress(self, step, loss, time_left):
        print(f'Step [{step}/{len(self.train_loader)}], Loss: {loss:.4f}, Time left: {time_left:.2f}s')

    def _save_checkpoint(self, epoch):
        torch.save(self.model.state_dict(), f'{self.checkpoint_path}/model_epoch_{epoch}.pth')
        print(f'Model checkpoint saved for epoch {epoch}')

    def _valid_iter(self, epoch):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images in self.valid_loader:
                images = images.to(self.device)

                # Assume src and tgt are obtained from images
                src = images
                tgt = images

                output = self.model(src, tgt)
                loss = self.criterion(output, images)

                running_loss += loss.item()

        avg_loss = running_loss / len(self.valid_loader)
        print(f'Validation Loss after epoch [{epoch + 1}/{self.num_epochs}]: {avg_loss:.4f}')
