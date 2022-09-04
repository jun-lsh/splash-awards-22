# you know i love copy pasting code
from typing import Tuple

import torch.nn as nn
import torch

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def conv_output_shape(d_h_w, kernel_size, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a tuple of (d,h,w) and returns a tuple of (d,h,w)
    """

    if type(d_h_w) is not tuple:
        d_h_w = (d_h_w, d_h_w, d_h_w)

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size, kernel_size)

    if type(stride) is not tuple:
        stride = (stride, stride, stride)

    if type(pad) is not tuple:
        pad = (pad, pad, pad)

    d = (d_h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1) // stride[0] + 1
    h = (d_h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1) // stride[1] + 1
    w = (d_h_w[2] + (2 * pad[2]) - (dilation * (kernel_size[2] - 1)) - 1) // stride[2] + 1

    return d, h, w


class GRUNet(nn.Module):
    """
    The input is expected to be a 5d tensor of dims [batch_size, time_steps, height, width, n_features]
    representing a height*width region for a certain time_steps period. The region would contain n_features of
    water_velocity, etc. to predict the final output. The final output will be 1 number representing the amount of
    microplastics pieces/m^3.
    """

    def __init__(self, input_dims: Tuple[int], hidden_size: int, output_dim=1, n_layers=1, drop_prob=0):
        super(GRUNet, self).__init__()

        self.batch_size, self.time_steps, self.height, self.width, self.n_features = input_dims

        # hidden size refers to the dimensions of the GRUs hidden state
        self.hidden_size = hidden_size
        # should be 1, as we are predicting only total microplastic concentration in that height*width region
        self.output_dim = output_dim
        # number of stacked GRUs (default 1)
        self.n_layers = n_layers

        # 1*3*3 kernel makes it independent of the time_steps dimension, but n_features being the channel makes sure the
        # convolutions depend on the value of the features of the surrounding pixels
        self.feature_kernel = (1, 3, 3)

        # series of convolutions to reduce dimensionality of height*width*n_features to height*width*1
        # Quite sure ts0 = time_steps as number of time steps should not be changed throughout the convolutions
        # [batch_size, n_features, time_steps, height, width] -> [batch_size, n_features, ts0, h0, w0]
        self.conv0 = nn.Conv3d(self.n_features, self.n_features, self.feature_kernel)
        ts0, h0, w0 = conv_output_shape((self.time_steps, self.height, self.width), self.feature_kernel)
        # TODO: Add more conv layers

        # this final conv layer will learn to compress n_features into 1 number, essentially a 1*1*1*n_features kernel
        # -> [batch_size, 1, ts0, h0, w0]
        self.feature_conv = nn.Conv3d(self.n_features, 1, (1,1,1))
        # squeeze out the 1 dim.
        # -> [batch_size, ts0, h0*w0] flattened final feature map of last 2 dims
        self.flatten = nn.Flatten(-2,-1)

        # input size of GRU will be the flattened feature map size = h0 * w0
        self.gru = nn.GRU(h0*w0, self.hidden_size, self.n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(self.hidden_size * ts0, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        # permute as conv3d accepts inputs of [batch_size, channels/n_features, D/time_steps, H, W]
        x = torch.permute(x, (0, 4, 1, 2, 3))
        x = self.conv0(x)
        x = self.feature_conv(x)
        x = torch.permute(x, (0, 2, 3, 4, 1))
        # remove the 1 dimensional n_features channel. new shape [batch_size, time_steps, height, width]
        x = torch.squeeze(x)
        x = self.flatten(x)

        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.gru.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

def training_loop(epochs: int, learning_rate: float, hidden_dim=256, EPOCHS=5):
    # Setting common hyperparameters, adjust later depending on input
    batch_size = 0
    time_steps = 0
    height = 0
    width = 0
    n_features = 0

    input_dim = (batch_size, time_steps, heigh, width, n_features)

    output_dim = 1
    n_layers = 1
    drop_prob = 0

    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers, drop_prob)
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print(f"Starting Training of model")
    epoch_times = []

    for i in range(epochs):
        start_time = time.time()
        h = model.init_hidden(batch_size)
        avg_loss = 0
        counter = 0


        # TODO: Implement dataloader to loop through, this wont work until we have that
        for x, label in train_loader:
            counter += 1
            h = h.data

            model.zero_grad()

            output, h = model.forward(x.to(device).float(), h)
            loss = criterion(output, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

            if counter%200 == 0:
                print(f"Epoch {epoch} | Step: {counter}/{len(train_loader)} | Average Loss for Epoch: {avg_loss/counter}")

        current_time = time.time()
        print(f"Epoch {epoch}/{EPOCHS} Done, Total Loss: {avg_loss/len(train_loader)}")
        print(f"Total Time Elapsed: {current_time-start_time} seconds")
        epoch_times.append(current_time-start_time)
    print(f"Total Training Time: {sum(epoch_times} seconds")

    return model
