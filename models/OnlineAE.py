import torch
import torch.nn as nn
import torch.optim as optim

class OnlineAE(nn.Module):
    def __init__(self, train_args):
        super().__init__()
        self.device = train_args["device"]    
        self.d = train_args["d"]
        self.L = train_args["L"]
        self.K = nn.Linear(self.L, self.L, bias=False)  # Learnable K matrix
        
        if "architecture" in train_args:
            layers_enc = []
            in_dim = self.d
            for hidden_dim in train_args["architecture"]:
                layers_enc.append(nn.Linear(in_dim, hidden_dim))
                layers_enc.append(nn.ReLU())
                in_dim = hidden_dim
            
            layers_enc.append(nn.Linear(in_dim, self.L))
            self.encoder = nn.Sequential(*layers_enc)
            
            # -------------------
            # Decoder
            # -------------------
            layers_dec = []
            in_dim = self.L
            for hidden_dim in reversed(train_args["architecture"]):
                layers_dec.append(nn.Linear(in_dim, hidden_dim))
                layers_dec.append(nn.ReLU())
                in_dim = hidden_dim
            
            layers_dec.append(nn.Linear(in_dim, self.d))
            self.decoder = nn.Sequential(*layers_dec)
            
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.d, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, self.L),
            )  
            
            self.decoder = nn.Sequential(
                nn.Linear(self.L, 8),
                nn.ReLU(),
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, self.d)
            )
        
        # training
        self.off_epochs = train_args["off_epochs"]
        self.max_iter = train_args["max_iter"]
        self.n_max = train_args["n_max"]
        self.lr = train_args["lr"]
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.parameters(), lr = self.lr)
        self.losses = []
        
        # spectral properties
        self.eigenvalues = None
        

    def forward(self, x, n_steps):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        K_power_n = torch.matrix_power(self.K.weight, n_steps)
        z_pred = z @ K_power_n.T
        x_pred = self.decoder(z_pred)
        return x_hat, z_pred, x_pred


    def _offline_training(self, X):
        '''
        one iteration
        (X is a array of shape (N,T,D): N trajectories, T time steps and D stands for dimension)
        '''
        x_target = X[:, -1, :]
        self.optimizer.zero_grad()
        
        loss = 0
        for tau in range(self.n_max):
            z_target = self.encoder(x_target)
            x = X[:, tau, :]
            x_hat, z_pred, x_pred = self.forward(x, self.n_max - tau)
            loss += self.criterion(x_pred, x_target) + self.criterion(x, x_hat) + self.criterion(z_pred, z_target)
            
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
    
    def offline_training(self, X_warm_up):
        X_pre = torch.tensor(X_warm_up[:,:-1,:].reshape(-1,self.d), dtype=torch.float32).to(self.device)
        X_after = torch.tensor(X_warm_up[:,1:,:].reshape(-1,self.d), dtype=torch.float32).to(self.device)
        
         
        for i in range(self.off_epochs):
            self.optimizer.zero_grad()
            Z_after = self.encoder(X_after)
            X_hat, Z_pred, X_pred = self.forward(X_pre, 1)
            loss = self.criterion(X_pred, X_after) + self.criterion(X_pre, X_hat) + self.criterion(Z_pred, Z_after)
            loss.backward()
            self.optimizer.step()
            if (i + 1) % 1000 == 0:
                print(f"Epoch {i+1}/{self.off_epochs}, Loss: {loss.item()}")
    
    def online_training(self, X_tilde):
        X_tilde = torch.tensor(X_tilde, dtype=torch.float32).to(self.device)
        for i in range(self.max_iter):
            self._offline_training(X_tilde)

    #### evaluation 
    def l_steps_error(self, X_tilde, l_step = 1):
        x_t_true = X_tilde[:, -1, :]
        x_t_l = X_tilde[:, -1 - l_step, :]
        _, _, x_t_pred = self.forward(x_t_l, l_step)     
        error = self.criterion(x_t_true, x_t_pred)
        return error.item()
    
    def compute_error(self, trajs):
        X_test = torch.tensor(trajs[:,:-1,:].reshape(-1,self.d), dtype=torch.float32).to(self.device)
        Y_test = torch.tensor(trajs[:,1:,:].reshape(-1,self.d), dtype=torch.float32).to(self.device)
        _, _, Y_pred = self.forward(X_test, 1)
        mse = ((Y_pred - Y_test) ** 2).mean().item()
        return mse
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        