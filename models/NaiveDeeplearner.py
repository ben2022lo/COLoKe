import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.linalg as la 

class Naive(nn.Module):
    def __init__(self, train_args):
        super().__init__()
        self.device = train_args["device"]    
        self.d = train_args["d"]
        self.L = train_args["L"]
        self.K = nn.Linear(self.d+self.L, self.d+self.L, bias=False)  # Learnable K matrix
        
        # model architecture
        if "architecture" in train_args:
            layers = []
            in_dim = self.d
            
            for hidden_dim in train_args["architecture"]:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            
            # Final layer (to self.L)
            layers.append(nn.Linear(in_dim, self.L))
            
            self.encoder = nn.Sequential(*layers)
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
        
        # training
        self.off_epochs = train_args["off_epochs"]
        self.max_iter = train_args["max_iter"]
        self.n_max = train_args["n_max"]
        self.lr = train_args["lr"]
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.parameters(), lr = self.lr)
        self.X_t_list = []
        self.X_t_next_list = []
        self.losses = []
        
        # spectral properties
        self.eigenvalues = None
        
    #### encoder and forward prediction in latent space ####
    def encode_z(self, x):
        '''
        encode with projection
        '''
        return torch.cat((x, self.encoder(x)), dim=1)
    
    def forward(self, x, n_steps):
        '''
        encode to latent space and predict n steps
        '''
        z = self.encode_z(x)
        K_power_n = torch.matrix_power(self.K.weight, n_steps)
        z_n_pred = z @ K_power_n.T

        return z_n_pred
    
    def offline_data(self, X):
        '''
        X is a array of shape (N,T,D): N trajectories, T time steps and D stands for dimension
        '''
        for l in range(1, self.n_max+1):
            self.X_t_list.append(X[:, :-l, :].reshape(-1, self.d))        # x_{t-l}
            self.X_t_next_list.append(X[:, l:, :].reshape(-1, self.d))    # x_t
        # Convert to PyTorch tensors
        self.X_t_list = [torch.tensor(x, dtype=torch.float32).to(self.device) for x in self.X_t_list]
        self.X_t_next_list = [torch.tensor(x, dtype=torch.float32).to(self.device) for x in self.X_t_next_list]
    
    def _offline_training(self):
        '''
        one iteration
        '''
        self.optimizer.zero_grad()
        loss = 0
        for l in range(1, self.n_max+1):
            z_pred = self.forward(self.X_t_list[l-1], l)
            z_true = self.encode_z(self.X_t_next_list[l-1])
            loss += self.criterion(z_pred, z_true)
            
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
        
    def offline_training(self, epochs):
        '''
        training on offline data
        '''
        for epoch in range(epochs):
            self._offline_training()
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {self.losses[-1]}")
                
    def get_eigenvalues(self, delta_t):
        K = self.K.weight.detach().cpu().numpy()
        eigenvalues, _ = la.eig(K.T)
        self.eigenvalues = np.log(eigenvalues) / delta_t
        
        return self.eigenvalues
        
    def eigen_functions(self, x):
        z = self.encode_z(x)       
        K = self.K.weight.detach().cpu().numpy()
        _, eigenvectors = la.eig(K.T)
        P = eigenvectors      
        evalu = P.T @ z.detach().cpu().numpy().T
        return evalu  
    
    def online_training(self, X_tilde, t):
        X_tilde = torch.tensor(X_tilde, dtype=torch.float32).to(self.device)
        self.offline_data(X_tilde) 
        for i in range(self.max_iter):
            self._offline_training()

    #### evaluation 
    def l_steps_error(self, X_tilde, l_step = 1):
        x_t_true = X_tilde[:, -1, :]
        x_t_l = X_tilde[:, -1 - l_step, :]
        z_t_pred = self.forward(x_t_l, l_step)    
        x_t_pred = z_t_pred[:,:self.d]    
        error = self.criterion(x_t_true, x_t_pred)
        return error.item()
    
    def compute_error(self, trajs):
        X_test = torch.tensor(trajs[:,:-1,:].reshape(-1,self.d), dtype=torch.float32).to(self.device)
        Y_test = torch.tensor(trajs[:,1:,:].reshape(-1,self.d), dtype=torch.float32).to(self.device)
        Y_pred = self.forward(X_test, 1)[:,:self.d]
        mse = ((Y_pred - Y_test) ** 2).mean().item()
        return mse
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        