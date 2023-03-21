import collections
import pathlib
import time
import typing

import torch
from torch.utils.data import DataLoader

import emojify.utils as utils


class Trainer:
    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        early_stop_count: int,
        epochs: int,
        model: torch.nn.Module,
        dataloaders: typing.List[DataLoader],
    ):
        """
        Initialize our trainer class.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_count = early_stop_count
        self.epochs = epochs

        # Since we are doing multi-class classification, we use CrossEntropyLoss
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        # Initialize the model
        self.model = model
        # Transfer model to GPU VRAM, if possible.
        self.model = utils.to_cuda(self.model)
        print(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.learning_rate)  # type: ignore # noqa

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = dataloaders

        # Validate our model everytime we pass through 50% of the dataset
        self.num_steps_per_val = len(self.dataloader_train) // 2
        self.global_step = 0
        self.start_time = time.time()

        # Tracking variables
        self.train_history = dict(
            loss=collections.OrderedDict(), accuracy=collections.OrderedDict()
        )
        self.validation_history = dict(
            loss=collections.OrderedDict(), accuracy=collections.OrderedDict()
        )
        self.checkpoint_dir = pathlib.Path("checkpoints")

    def validation_step(self):
        """
        Computes the loss/accuracy for all three datasets.
        Train, validation and test.
        """
        self.model.eval()  # type: ignore
        # validation_loss, validation_acc = compute_loss_and_accuracy(
        #    self.dataloader_val, self.model, self.loss_criterion
        # )
        # self.validation_history["loss"][self.global_step] = validation_loss
        # self.validation_history["accuracy"][self.global_step] = validation_acc
        used_time = time.time() - self.start_time
        print(
            f"Epoch: {self.epoch:>1}",
            f"Batches per seconds: {self.global_step / used_time:.2f}",
            f"Global step: {self.global_step:>6}",
            # f"Validation Loss: {validation_loss:.2f}",
            # f"Validation Accuracy: {validation_acc:.3f}",
            sep=", ",
        )
        self.model.train()  # type: ignore

    def should_early_stop(self):
        """
        Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss
        # list.
        val_loss = self.validation_history["loss"]
        if len(val_loss) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = list(val_loss.values())[-self.early_stop_count :]
        first_loss = relevant_loss[0]
        if first_loss == min(relevant_loss):
            print("Early stop criteria met")
            return True
        return False

    def train_step(self, X_batch, Y_batch):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the
        train step.
        The function returns the mean loss value which is then automatically logged in
        our variable self.train_history.
        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32]
        # Y_batch is the CIFAR10 image label. Shape: [batch_size]
        # Transfer images / labels to GPU VRAM, if possible
        X_batch = utils.to_cuda(X_batch)
        Y_batch = utils.to_cuda(Y_batch)

        # Perform the forward pass
        predictions = self.model(X_batch)  # type: ignore
        # Compute the cross entropy loss for the batch
        loss = self.loss_criterion(predictions, Y_batch)
        # Backpropagation
        loss.backward()
        # Gradient descent step
        self.optimizer.step()
        # Reset all computed gradients to 0
        self.optimizer.zero_grad()

        return loss.detach().cpu().item()

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """

        def should_validate_model():
            return self.global_step % self.num_steps_per_val == 0

        for epoch in range(self.epochs):
            self.epoch = epoch
            # Perform a full pass through all the training samples
            for X_batch, Y_batch in self.dataloader_train:
                loss = self.train_step(X_batch, Y_batch)
                self.train_history["loss"][self.global_step] = loss
                self.global_step += 1
                # Compute loss/accuracy for validation set
                if should_validate_model():
                    self.validation_step()
                    self.save_model()
                    if self.should_early_stop():
                        print("Early stopping.")
                        return

    def save_model(self):
        def is_best_model():
            """
            Returns True if current model has the lowest validation loss
            """
            val_loss = self.validation_history["loss"]
            validation_losses = list(val_loss.values())
            return validation_losses[-1] == min(validation_losses)

        state_dict = self.model.state_dict()  # type: ignore
        filepath = self.checkpoint_dir.joinpath(f"{self.global_step}.ckpt")

        utils.save_checkpoint(state_dict, filepath, is_best_model())

    def load_best_model(self):
        state_dict = utils.load_best_checkpoint(self.checkpoint_dir)
        if state_dict is None:
            print(
                f"Could not load best checkpoint. Did not find under: {self.checkpoint_dir}"  # type: ignore # noqa
            )
            return
        self.model.load_state_dict(state_dict)  # type: ignore
