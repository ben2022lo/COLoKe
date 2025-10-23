import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.linalg as la 


class CKL(nn.Module):
    def __init__(self, train_args, CP_args):
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
        self.iter_t = []
        self.test_error_evolution = []
        
        # conformal PI
        self.scores = []
        self.scores_after = []
        self.qs = []
        self.gs = []
        self.alpha = CP_args["alpha"]
        self.eta = CP_args["eta"]
        self.Csat = CP_args["Csat"]
        self.T_warm_up = CP_args["T_warm_up"]
        self.B_hat_horizon = self.T_warm_up-self.n_max
           
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
        
        
    def offline_training(self, epochs, trajs_test=None):
        '''
        training on offline data
        '''
        for epoch in range(epochs):
            self._offline_training()
            
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {self.losses[-1]}")
    
    def online_training(self, X_tilde, t):
        X_tilde = torch.tensor(X_tilde, dtype=torch.float32).to(self.device)
        s_t = self.score_function(X_tilde) # calculate s_t
        self.scores.append(s_t)
        q_t = self.conformalPI(t) # update q_t+1 and return q_t
        
        it = 0
        
        s_t_true = s_t
        if s_t_true >= q_t:
            self.offline_data(X_tilde)
            while s_t >= q_t and it <= self.max_iter:
                self._offline_training()
                s_t = self.score_function(X_tilde)
                it += 1
        else:
            for _ in range(1):
                self.offline_data(X_tilde)
            it = 1
        self.iter_t.append(it)
        self.scores_after.append(s_t)
        return False if it == 1 else True
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
    
    def l_steps_error(self, X_tilde, l_step = 1):
        x_t_true = X_tilde[:, -1, :]
        x_t_l = X_tilde[:, -1 - l_step, :]
        z_t_pred = self.forward(x_t_l, l_step)    
        x_t_pred = z_t_pred[:,:self.d]    
        error = self.criterion(x_t_true, x_t_pred)
        return error.item()
        
    #### score function and Conformal PI procedure ####
    def score_function(self, X_tilde):
        score = 0.0
        
        x_t = X_tilde[:, -1, :]
        z_t_true = self.encode_z(x_t)
        
        for l in range(1, self.n_max + 1):
            x_t_l = X_tilde[:, self.n_max - l, :]
            z_t_pred = self.forward(x_t_l, l)    
            score += self.criterion(z_t_pred, z_t_true)
        
        return score.item()
    
    #### conformal PI 
    def mytan(self, x):
        if x >= np.pi/2:
            return np.infty
        elif x <= -np.pi/2:
            return -np.infty
        else:
            return np.tan(x)
    
    def saturation_fn_log(self, x, t, KI):
        if KI == 0:
            return 0
        tan_out = self.mytan(x * np.log(t+1)/(self.Csat * (t+1)))
        out = KI * tan_out
        return  out
        
    def conformalPI(self, t):
        s_t = self.scores[-1]
        q_t = self.qs[-1]
        err_t = 1 if s_t > q_t else 0
        B_hat = np.max(self.scores[-self.n_max-1:-1]) 
             
        x = np.sum(self.gs)
        g_t = err_t - self.alpha 
        self.gs.append(g_t)
        
        q_t_next = q_t + self.eta * B_hat * g_t + self.saturation_fn_log(x, t, B_hat)
        if q_t_next <= 0:
            q_t_next = q_t
        self.qs.append(q_t_next)
        
        return q_t
    
    def initialization(self, X_warm_up):
        X_warm_up = torch.tensor(X_warm_up, dtype=torch.float32).to(self.device)
        for t in range(self.n_max+1, self.T_warm_up):
            X_tilde = X_warm_up[:, t-self.n_max : t+1, :]
            self.scores.append(self.score_function(X_tilde))
        self.qs.append(np.quantile(self.scores, 1-self.alpha))
        
    #### evaluation 
    def compute_error(self, trajs):
        X_test = torch.tensor(trajs[:,:-1,:].reshape(-1,self.d), dtype=torch.float32).to(self.device)
        Y_test = torch.tensor(trajs[:,1:,:].reshape(-1,self.d), dtype=torch.float32).to(self.device)
        Y_pred = self.forward(X_test, 1)[:,:self.d]
        mse = ((Y_pred - Y_test) ** 2).mean().item()
        return mse
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        