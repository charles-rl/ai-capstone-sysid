import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BaseModel(nn.Module):
    def __init__(self, chkpt_file_pth, device):
        super(BaseModel, self).__init__()

        self.chkpt_file_pth = chkpt_file_pth
        self.loss = nn.GaussianNLLLoss(full=False)

        self.device = device

    def forward(self, data):
        return NotImplementedError

    def learn(self, data, target):
        data = data.to(self.device)
        target = target.to(self.device)
        mu, sigma = self.forward(data)
        # if sigma then square here else if var then do not square
        loss = self.loss(mu, target, sigma.pow(2))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clip_value)
        self.optimizer.step()
        return loss.item()

    def save_model(self):
        print("...saving checkpoint...")
        torch.save({"model": self.state_dict()}, self.chkpt_file_pth)

    def load_model(self):
        print("...loading checkpoint...")
        checkpoint = torch.load(self.chkpt_file_pth, map_location=self.device)
        self.load_state_dict(checkpoint["model"])


class CNNLSTMModel(BaseModel):
    def __init__(self, config: dict, n_params, chkpt_file_pth, device):
        super(CNNLSTMModel, self).__init__(chkpt_file_pth, device)
        
        in_channels = config["in_channels"]
        lr = float(config["learning_rate"])
        cnn1_dims = config["cnn1_dims"]
        cnn2_dims = config["cnn2_dims"]
        lstm_dims = config["lstm_dims"]
        weight_decay = float(config["weight_decay"])
        self.clip_value = config["clip_value"]
        
        self.cnn1_low_freq = nn.Conv1d(
            in_channels=in_channels, out_channels=cnn1_dims, kernel_size=7, dilation=4, padding="same"
        )
        self.cnn1_mid_freq = nn.Conv1d(
            in_channels=in_channels, out_channels=cnn1_dims, kernel_size=5, dilation=2, padding="same"
        )
        self.cnn1_high_freq = nn.Conv1d(
            in_channels=in_channels, out_channels=cnn1_dims, kernel_size=3, padding="same"
        )
        self.bn1_low_freq = nn.BatchNorm1d(cnn1_dims)
        self.bn1_mid_freq = nn.BatchNorm1d(cnn1_dims)
        self.bn1_high_freq = nn.BatchNorm1d(cnn1_dims)
        
        # Concatenating the outputs of the 3 CNNs so cnn1_dims * 3
        self.cnn2 = nn.Conv1d(
            in_channels=cnn1_dims * 3, out_channels=cnn2_dims, kernel_size=3, padding="same"
        )
        self.bn2 = nn.BatchNorm1d(cnn2_dims)
        
        # Downsampling to about half the sequence length so that the LSTM sees about 300 timesteps instead of 600 timesteps
        # Which reduces the vanishing gradient
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.lstm = nn.LSTM(
            input_size=cnn2_dims,
            hidden_size=lstm_dims,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.ln_lstm = nn.LayerNorm(lstm_dims * 2)
        
        self.fc = nn.Linear(lstm_dims * 2, lstm_dims)
        self.ln_fc = nn.LayerNorm(lstm_dims)
        
        self.mu_fc = nn.Linear(lstm_dims, n_params)
        self.sigma_fc = nn.Linear(lstm_dims, n_params)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.to(self.device)
        
    def forward(self, x: torch.tensor):
        y_low = self.bn1_low_freq(F.mish(self.cnn1_low_freq(x)))
        y_mid = self.bn1_mid_freq(F.mish(self.cnn1_mid_freq(x)))
        y_high = self.bn1_high_freq(F.mish(self.cnn1_high_freq(x)))
        
        # We want it to be dim=1 because the shape becomes (batch_size, cnn1_dims * 3, timesteps)
        # if dim=-1 then shape becomes (batch_size, cnn1_dims, timesteps * 3)
        y = torch.cat([y_low, y_mid, y_high], dim=1)
        y = self.bn2(F.mish(self.cnn2(y)))
        
        y = self.pool(y)
        
        # swap in_channels and sequence_length
        y = y.permute(0, 2, 1)
        
        output_lstm, (h_n, c_n) = self.lstm(y)
        
        # we only want the h_n
        # flatten
        last_layer_forward_h = h_n[-2, :, :]  # Shape: (batch_size, lstm_dim)
        last_layer_backward_h = h_n[-1, :, :]  # Shape: (batch_size, lstm_dim)
        y = torch.cat((last_layer_forward_h, last_layer_backward_h), dim=1)
        y = self.ln_lstm(F.mish(y))
        
        y = self.ln_fc(F.mish(self.fc(y)))
        
        mu = torch.tanh(self.mu_fc(y)) * 1.2  # Scale the tanh to allow for OOD predictions
        sigma = torch.exp(self.sigma_fc(y)) + 1e-6  # epsilon added to help with the stability when the sigma is near 0
        # sigma = standard deviation
        # sigma^2 = variance = var
        # upon testing exponential and sigma works better
        return mu, sigma
