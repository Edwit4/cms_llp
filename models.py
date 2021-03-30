import torch
from torch import nn

class image_net(nn.Module):
    def __init__(self, channels):
        conv_size = 8
        linear_size = 100
        image_dim = 32
        num_max_pool = 0
        k_size = 3
        stride = 1
        pad_size = ((stride-1)*image_dim-stride+k_size)//2
        linear_in = (image_dim // (2**num_max_pool))**2 * conv_size
        super(image_net, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(channels, conv_size, kernel_size=k_size,
                    stride=stride, padding=pad_size),
                nn.LeakyReLU())

        self.layer2 = nn.Sequential(
                nn.Conv2d(conv_size, conv_size, kernel_size=k_size,
                    stride=stride, padding=pad_size),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
                nn.Conv2d(conv_size, conv_size, kernel_size=k_size,
                    stride=stride, padding=pad_size),
                nn.LeakyReLU())

        self.layer4 = nn.Sequential(
                nn.Conv2d(conv_size, conv_size, kernel_size=k_size,
                    stride=stride, padding=pad_size),
                nn.LeakyReLU())
                #nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(linear_in, linear_size)
        self.relu1 = nn.LeakyReLU()
        self.dropout_1 = nn.Dropout()
        self.fc2 = nn.Linear(linear_size, linear_size)
        self.relu2 = nn.LeakyReLU()
        self.dropout_2 = nn.Dropout()
        self.fc3 = nn.Linear(linear_size, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        #out = self.layer2(out)
        #out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        #out = self.dropout_1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        #out = self.dropout_2(out)
        out = self.fc3(out)
        out = self.sig(out)
        return out

class isoefp_net(nn.Module):
    def __init__(self, num_cones, num_efps):
        super(isoefp_net, self).__init__()
        layer_size = 256
        drop_prob = 0.2
        self.input_layer = nn.Linear(num_cones + num_efps, layer_size)
        self.hidden_fc_1 = nn.Linear(layer_size,layer_size)
        self.relu1 = nn.LeakyReLU()
        self.dropout_1 = nn.Dropout(p=drop_prob)
        self.hidden_fc_2 = nn.Linear(layer_size,layer_size)
        self.relu2 = nn.LeakyReLU()
        self.dropout_2 = nn.Dropout(p=drop_prob)
        self.hidden_fc_3 = nn.Linear(layer_size,layer_size)
        self.relu3 = nn.LeakyReLU()
        self.dropout_3 = nn.Dropout(p=drop_prob)
        self.hidden_fc_4 = nn.Linear(layer_size,layer_size)
        self.relu4 = nn.LeakyReLU()
        self.out_layer = nn.Linear(layer_size,1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.input_layer(x)
        out = self.hidden_fc_1(out)
        out = self.relu1(out)
        #out = self.dropout_1(out)
        out = self.hidden_fc_2(out)
        out = self.relu2(out)
        #out = self.dropout_2(out)
        out = self.hidden_fc_3(out)
        out = self.relu3(out)
        #out = self.dropout_3(out)
        out = self.hidden_fc_4(out)
        out = self.relu4(out)
        out = self.out_layer(out)
        out = self.sig(out)
        return out

class PFN(nn.Module):
    def __init__(self, Phi_size, F_size, l):
        super(PFN, self).__init__()

        self.phi_input = nn.Linear(3, Phi_size)
        self.phi_fc1 = nn.Linear(Phi_size, Phi_size)
        self.phi_relu1 = nn.LeakyReLU()
        self.dropout_1 = nn.Dropout(p=0.2)
        self.phi_fc2 = nn.Linear(Phi_size, Phi_size)
        self.phi_relu2 = nn.LeakyReLU()
        self.dropout_2 = nn.Dropout(p=0.2)
        self.phi_out = nn.Linear(Phi_size, l)

        self.F_input = nn.Linear(l, F_size)
        self.F_fc1 = nn.Linear(F_size, F_size)
        self.F_relu1 = nn.LeakyReLU()
        self.F_fc2 = nn.Linear(F_size, F_size)
        self.F_relu2 = nn.LeakyReLU()
        self.F_fc3 = nn.Linear(F_size, F_size)
        self.F_relu3 = nn.LeakyReLU()
        self.F_out = nn.Linear(F_size, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.phi_input(x)
        out = self.phi_fc1(out)
        out = self.phi_relu1(out)
        out = self.phi_fc2(out)
        out = self.phi_relu2(out)
        out = self.phi_out(out)

        out = torch.sum(out, dim=1)

        out = self.F_input(out)
        out = self.F_fc1(out)
        out = self.F_relu1(out)
        out = self.F_fc2(out)
        out = self.F_relu2(out)
        out = self.F_fc3(out)
        out = self.F_relu3(out)
        out = self.F_out(out)
        out = self.sig(out)

        return out

class PFN_weights(nn.Module):
    def __init__(self, Phi_size, F_size, l):
        super(PFN_weights, self).__init__()

        self.phi_input = nn.Linear(4, Phi_size)
        self.phi_fc1 = nn.Linear(Phi_size, Phi_size)
        self.phi_relu1 = nn.LeakyReLU()
        self.phi_fc2 = nn.Linear(Phi_size, Phi_size)
        self.phi_relu2 = nn.LeakyReLU()
        self.phi_out = nn.Linear(Phi_size, l)

        self.F_input = nn.Linear(l, F_size)
        self.F_fc1 = nn.Linear(F_size, F_size)
        self.F_relu1 = nn.LeakyReLU()
        self.F_fc2 = nn.Linear(F_size, F_size)
        self.F_relu2 = nn.LeakyReLU()
        self.F_fc3 = nn.Linear(F_size, F_size)
        self.F_relu3 = nn.LeakyReLU()
        self.F_out = nn.Linear(F_size, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.phi_input(x)
        out = self.phi_fc1(out)
        out = self.phi_relu1(out)
        out = self.phi_fc2(out)
        out = self.phi_relu2(out)
        out = self.phi_out(out)

        out = torch.sum(out, dim=1)

        out = self.F_input(out)
        out = self.F_fc1(out)
        out = self.F_relu1(out)
        out = self.F_fc2(out)
        out = self.F_relu2(out)
        out = self.F_fc3(out)
        out = self.F_relu3(out)
        out = self.F_out(out)
        out = self.sig(out)

        return out
