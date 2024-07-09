import torch
import os
import time
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FontTrainer:
    def __init__(self,
                 model, train_loader, valid_loader, criterion, optimizer, device, train_conf, data_conf, ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_conf = train_conf
        self.data_conf = data_conf
        self.best_loss = float('inf')

    def train(self):
        num_epochs = self.train_conf['num_epochs']
        logger.info(f"start training iterations: {num_epochs}")

        train_loader_iter = iter(self.train_loader)
        print(train_loader_iter)
        start_time = time.time()
        for epoch in range(num_epochs):
            try:
                data = next(train_loader_iter)
            except Exception as e:
                logger.error(f"Error: {e}\ntrain_loader_iter:\n{train_loader_iter}")
                return
            print(epoch)
            print(data)
            # self._train_iter(data, epoch)
            # self._save_checkpoint(epoch)
            # self._valid_iter(epoch)
            # self._progress(epoch, num_epochs, start_time)

    def _train_iter(self, data, epoch):
        self.model.train()
        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        logger.info(f"Epoch {epoch}, Train Loss: {loss.item()}")

    def _valid_iter(self, epoch):
        self.model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for data in self.valid_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                valid_loss += loss.item()

        valid_loss /= len(self.valid_loader)
        logger.info(f"Epoch {epoch}, Validation Loss: {valid_loss}")

        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self._save_best_model(epoch, valid_loss)

    def _save_checkpoint(self, epoch):
        if epoch >= self.train_conf['SNAPSHOT_BEGIN'] and epoch % self.train_conf['SNAPSHOT_EPOCH'] == 0:
            checkpoint_path = os.path.join(self.data_conf['save_model_dir'], f'checkpoint_epoch_{epoch}.pt')
            model_state_dict = self.model.module.state_dict() if isinstance(self.model,
                                                                            torch.nn.DataParallel) else self.model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.criterion
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

    def _save_best_model(self, epoch, loss):
        best_model_path = os.path.join(self.data_conf['save_model_dir'], 'best_model.pt')
        model_state_dict = self.model.module.state_dict() if isinstance(self.model,
                                                                        torch.nn.DataParallel) else self.model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }, best_model_path)
        logger.info(f"Best model saved at epoch {epoch} with validation loss {loss} to {best_model_path}")

    def _progress(self, epoch, num_epochs, start_time):
        elapsed_time = time.time() - start_time
        epochs_left = num_epochs - epoch - 1
        eta = elapsed_time / (epoch + 1) * epochs_left
        eta_minutes, eta_seconds = divmod(eta, 60)
        logger.info(f"Epoch {epoch}/{num_epochs} completed. ETA: {int(eta_minutes)} minutes {int(eta_seconds)} seconds")
