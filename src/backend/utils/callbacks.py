import keras.api.callbacks
import matplotlib.pyplot as plt
import numpy as np
from keras.api.callbacks import LambdaCallback


class LearningRateTracker(keras.callbacks.Callback):
    def __init__(self, file_path="learning-rate-plot.png"):
        super().__init__()
        self.learning_rates = []
        self.file_path = file_path

    def on_epoch_end(self, epoch, logs=None):
        opt = self.model.optimizer

        if hasattr(opt, "inner_optimizer"):
            opt = opt.inner_optimizer

        if callable(opt.learning_rate):
            lr = opt.learning_rate(opt.iterations).numpy()
        else:
            lr = opt.learning_rate

        self.learning_rates.append(lr)
        print(f"Epoch {epoch + 1}: Learning rate = {lr}")

    def on_train_end(self, logs=None):
        plt.plot(range(1, len(self.learning_rates) + 1), self.learning_rates, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.savefig(self.file_path)


class LearningRateFinder:
    def __init__(self, model, stop_factor=4, beta=0.98):
        self.model = model
        self.stop_factor = stop_factor
        self.beta = beta
        self.lrs = []
        self.losses = []
        self.lr_mult = 1
        self.avg_loss = 0
        self.best_loss = np.inf
        self.batch_num = 0

    def reset(self):
        """Reset variables for a new learning rate search."""
        self.lrs = []
        self.losses = []
        self.lr_mult = 1
        self.avg_loss = 0
        self.best_loss = np.inf
        self.batch_num = 0

    def on_batch_end(self, batch, logs):
        """Callback to update learning rate and track loss."""
        optimizer = self.model.optimizer
        lr = optimizer.learning_rate.numpy()
        self.lrs.append(lr)

        loss = logs["loss"]
        self.batch_num += 1

        # Compute smoothed loss
        self.avg_loss = (self.beta * self.avg_loss) + ((1 - self.beta) * loss)
        smooth_loss = self.avg_loss / (1 - (self.beta**self.batch_num))
        self.losses.append(smooth_loss)

        # Check if loss exceeds the stopping threshold
        stop_loss = self.stop_factor * self.best_loss
        if self.batch_num > 1 and smooth_loss > stop_loss:
            self.model.stop_training = True

        # Update best loss if necessary
        if smooth_loss < self.best_loss or self.batch_num == 1:
            self.best_loss = smooth_loss

        # Increase learning rate
        lr *= self.lr_mult
        optimizer.learning_rate.assign(lr)

    def find(
        self, train_dataset, start_lr=1e-7, end_lr=10, epochs=1, steps_per_epoch=None, verbose=1
    ):
        """Perform the learning rate range test."""
        # Reset state
        self.reset()

        # Calculate total number of updates and learning rate multiplier
        num_updates = epochs * steps_per_epoch if steps_per_epoch else epochs * len(train_dataset)
        self.lr_mult = (end_lr / start_lr) ** (1 / num_updates)

        # Save initial weights to restore later
        initial_weights = self.model.get_weights()

        # Set initial learning rate
        optimizer = keras.api.optimizers.Adam(learning_rate=start_lr)
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy")

        # Create callback for updating learning rate after each batch
        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        # Train the model with the callback to adjust learning rates dynamically
        try:
            self.model.fit(
                train_dataset,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                verbose=verbose,
                callbacks=[callback],
            )
        finally:
            # Restore original weights and learning rate after training ends
            self.model.set_weights(initial_weights)
            optimizer.learning_rate.assign(start_lr)

    def plot_loss(self, skip_begin=10, skip_end=1):
        """Plot the loss as a function of the learning rate."""
        lrs = self.lrs[skip_begin:-skip_end]
        losses = self.losses[skip_begin:-skip_end]

        plt.figure(figsize=(8, 6))
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Loss")
        plt.title("Learning Rate Finder")
        plt.show()
