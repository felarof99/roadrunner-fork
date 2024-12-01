"""neuron_parallel_compile torchrun --nproc_per_node=2 src/felafax/trainer_engine/trainer_mnist.py"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm

# Global constants
EPOCHS = 4
WARMUP_STEPS = 2
BATCH_SIZE = 32


# Create dummy MNIST data
def create_dummy_mnist_data(num_samples=1000):
    # MNIST images are 28x28 pixels, labels are 0-9
    dummy_data = torch.randn(num_samples, 1, 28, 28)
    dummy_labels = torch.randint(0, 10, (num_samples,))
    return dummy_data, dummy_labels


class DummyDataLoader:
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples = len(data)
        self.current_idx = 0

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration

        end_idx = min(self.current_idx + self.batch_size, self.num_samples)
        batch_data = self.data[self.current_idx : end_idx]
        batch_labels = self.labels[self.current_idx : end_idx]
        self.current_idx = end_idx

        return batch_data, batch_labels

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


# Declare 3-layer MLP for MNIST dataset
class MLP(nn.Module):
    def __init__(self, input_size=28 * 28, output_size=10, layers=[120, 84]):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def main():
    # Create dummy data
    dummy_data, dummy_labels = create_dummy_mnist_data()

    # Create dummy data loader
    train_loader = DummyDataLoader(dummy_data, dummy_labels, BATCH_SIZE)

    # Fix the random number generator seeds for reproducibility
    torch.manual_seed(0)

    # XLA: Specify XLA device (defaults to a NeuronCore on Trn1 instance)
    device = "xla"

    # Move model to device and declare optimizer and loss function
    model = MLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.NLLLoss()

    # Run the training loop
    print("----------Training ---------------")
    model.train()
    for epoch in range(EPOCHS):
        start = time.time()
        for idx, (train_x, train_label) in enumerate(train_loader):
            optimizer.zero_grad()
            train_x = train_x.view(train_x.size(0), -1)
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            output = model(train_x)
            loss = loss_fn(output, train_label)
            loss.backward()
            optimizer.step()
            xm.mark_step()  # XLA: collect ops and run them in XLA runtime
            if idx < WARMUP_STEPS:  # skip warmup iterations
                start = time.time()

        # Compute statistics for the last epoch
        interval = idx - WARMUP_STEPS  # skip warmup iterations
        throughput = interval / (time.time() - start)
        print(f"Train throughput (iter/sec): {throughput}")
        print(f"Final loss is {loss.detach().to('cpu'):0.4f}")

    # Save checkpoint for evaluation
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {"state_dict": model.state_dict()}
    xm.save(checkpoint, "checkpoints/checkpoint.pt")

    print("----------End Training ---------------")


if __name__ == "__main__":
    main()
