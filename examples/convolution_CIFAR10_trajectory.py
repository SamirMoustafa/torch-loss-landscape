from copy import deepcopy
from os.path import join, exists
from pathlib import Path

from torch import cuda, load, save
from torch import device as torch_device
from torch import no_grad
from torch.nn import Conv2d, Linear, ReLU, Sequential, Flatten
from torch.nn.functional import cross_entropy
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from torch_landscape import plot_loss_landscape_trajectory
from torch_landscape.utils import clone_parameters, seed_everything


# Store parameter trajectory
parameters_with_loss = []

# Define the training loop
def train(model, device, train_loader, optimizer, epoch):
    intermediate_parameters = []
    losses, accuracies = [], []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        accuracies += [pred.eq(target.view_as(pred)).sum().item() / len(data)]
        losses += [loss.item()]

        if batch_idx % 100 == 0:
            intermediate_parameters.append((clone_parameters(model.parameters()), loss.item()))
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100.0 * batch_idx / len(train_loader),
                                                                           loss.item()))
    # Store a snapshot every 2 epochs
    parameters_with_loss.append((clone_parameters(model.parameters()), loss.item()))

    print("Train Epoch: {}\tAccuracy: {:.6f}".format(epoch, sum(accuracies) / len(accuracies)))
    return intermediate_parameters


# Define the evaluate loop
def evaluate():
    eval_device = [*model.parameters()][0].device
    model.eval()
    test_loss = 0
    correct = 0
    with no_grad():
        for data, target in test_loader:
            data, target = data.to(eval_device), target.to(eval_device)
            output = model(data)
            test_loss += cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    return test_loss


if __name__ == "__main__":
    epochs = 20
    lr = 0.01

    # Set up the device (GPU or CPU), data paths, and checkpoint directories
    device = torch_device("cuda" if cuda.is_available() else "cpu")
    data_path = "./data"
    checkpoint_dir =  ["./"] + ["checkpoints_with_trajectory"]
    if not exists(join(*checkpoint_dir)):
        Path(join(*checkpoint_dir)).mkdir(parents=True, exist_ok=True)

    # Load the CIFAR10 dataset
    transform = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset = CIFAR10(data_path, train=True, download=True, transform=transform)
    test_dataset = CIFAR10(data_path, train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

    # Initialize the network and optimizer
    # Define Two Convolutional layers followed by a Linear layer
    model = Sequential(
        Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        ReLU(),
        Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        ReLU(),
        Flatten(),
        Linear(64 * 32 * 32, 512),
        ReLU(),
        Linear(512, 10)
    ).to(device)
    file_name = ["{}_{}".format(train_dataset.__class__.__name__,
                                   model.__class__.__name__)]
    checkpoint_file_dir = join(*checkpoint_dir + ["convolution_cifar10.pt"])
    print(checkpoint_file_dir)
    if exists(checkpoint_file_dir):
        state_dict = load(checkpoint_file_dir)["model_state_dict"]
        parameters_with_loss = load(checkpoint_file_dir)["parameters_with_loss"]
        model.load_state_dict(state_dict)
    else:
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        best_loss = float("inf")
        best_model_state_dict = None
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            test_loss = evaluate()
            if test_loss < best_loss:
                best_loss = test_loss
                best_model_state_dict = deepcopy(model.state_dict())
        save({"model_state_dict": best_model_state_dict,
              "parameters_with_loss": parameters_with_loss}, checkpoint_file_dir)

    print(f"Plotting the loss trajectory for {file_name}")
    plot_loss_landscape_trajectory(optimal_parameters=model.parameters(),
                                   parameters_snapshot_with_respective_loss=parameters_with_loss,
                                   model=model,
                                   evaluate_function=evaluate,
                                   file_directory="Convolution_CIFAR10_trajectory_plot")