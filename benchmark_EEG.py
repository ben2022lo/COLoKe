import numpy as np
from models import CKL, Naive, OnlineEDMD, OnlineAE, Phi_poly
from odmd import OnlineDMD
from sklearn.preprocessing import MinMaxScaler
import torch
import time
import mne
import matplotlib.pyplot as plt
#### using gpu for deep models
session_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(session_device)
print("device: ",session_device)
torch.manual_seed(0)


# Load EDF EEG data
raw = mne.io.read_raw_edf("data/real_EEG/S001R01.edf", preload=True)
data, times = raw.get_data(return_times=True)
channels, time_steps = data.shape  
# downsampling
new_T = 976  
window_size = time_steps // new_T  # 9760 // 976 = 10
data_subsampled = data.reshape(channels, new_T, window_size).mean(axis=2)
scaler = MinMaxScaler(feature_range=(0, 1))  # scale each channel to [0, 1]
data_scaled = scaler.fit_transform(data_subsampled.T)

# Plot normalized trajectories
plt.figure(figsize=(12, 8))
for ch in range(data_scaled.shape[1]):  # loop over 64 channels
    plt.plot(data_scaled[:, ch], label=f"Ch {ch}")  
plt.title("64 EEG channel trajectories")
plt.xlabel("Time steps")
plt.ylabel("Normalized Amplitude")
plt.tight_layout()
plt.show()

# Add batch dimension for Koopman learner
traj = np.expand_dims(data_scaled[:500,:], axis=0)  # shape = (1, T, d)

T = traj.shape[1]
d = traj.shape[2]
T_warm_up = 100

results = {}

########## COLoKe ##########
# training hyper-parameters
train_args = {
    "d" : d,
    "L" : 1,   # extended dictionary 
    "n_max" : 1, # max prediction horizon
    "off_epochs" : int(5e2),
    "max_iter" : int(5),
    "architecture": [64],
    "lr" : 1e-3,
    "device" : session_device
    }
# Conformal PI hyper-parameters
CP_args = {
    "alpha" : 0.5,
    "eta" : 0.1,
    "Csat" : 20,
    "T_warm_up" : T_warm_up}

start = time.time()
# warm up with avalable data
print("-----warm up started-----")
X_warm_up = traj[:, :CP_args["T_warm_up"], :]
model = CKL(train_args, CP_args).to(session_device)
model.offline_data(X_warm_up)
model.offline_training(model.off_epochs)
print("-----warm up finished-----")

# Initialize q
model.initialization(X_warm_up)
# online training
print("-----online training started-----")
online_mse = np.empty(T-1-T_warm_up)
for t in range(model.T_warm_up + 1, T):
    X_tilde = traj[:, t-model.n_max : t+1, :]
    # get online error
    online_mse[t-1-T_warm_up] = model.l_steps_error(torch.tensor(X_tilde, dtype=torch.float32).to(model.device))
    # update the model
    model.online_training(X_tilde, t)   
print("-----online training finished-----")
end = time.time()
results["COLoKe"] = {}
results["COLoKe"]["mean"] = np.mean(online_mse)
results["COLoKe"]["std"] = np.std(online_mse)
results["COLoKe"]["time"] = end-start


########## OnlineDMD ##########

X_train = traj[:,:-1,:]
Y_train = traj[:,1:,:]
# initialization
X_warm_up = X_train[:,:T_warm_up,:].reshape(-1,d).T
Y_warm_up = Y_train[:,:T_warm_up,:].reshape(-1,d).T

start = time.time()
model = OnlineDMD(d, 1)
model.initialize(X_warm_up, Y_warm_up)

# online training
online_mse = np.empty(T-1-T_warm_up) 
for t in range(T_warm_up, T-1):
    # get online mse
    mse = ((Y_train[0,t,:].T - model.A @ X_train[0,t,:].T) ** 2).mean().item()
    online_mse[t-T_warm_up] = mse
    # update the model
    model.update(X_train[0,t,:].T, Y_train[0,t,:].T)
end = time.time()

# storage
results["ODMD"] = {}
results["ODMD"]["mean"] = np.mean(online_mse)
results["ODMD"]["std"] = np.std(online_mse)
results["ODMD"]["time"] = end-start

########## OnlineAE ##########
train_args = {
    "d" : d,
    "L" : 32,   # extended dictionary 
    "n_max" : 1, # max prediction horizon
    "off_epochs" : int(5e2),
    "max_iter" : int(5),
    "lr" : 1e-3,
    "architecture": [64],
    "device" : session_device
    }
start = time.time()
# warm up with avalable data
print("-----warm up started-----")
X_warm_up = traj[:, :T_warm_up, :]
model = OnlineAE(train_args).to(session_device)
model.offline_training(X_warm_up)
print("-----warm up finished-----")

# online training
online_mse = np.empty(T-1-T_warm_up)
print("-----online training started-----")
for t in range(T_warm_up + 1, T):
    X_tilde = traj[:, t-model.n_max : t+1, :]
    # get online error
    online_mse[t-1-T_warm_up] = model.l_steps_error(torch.tensor(X_tilde, dtype=torch.float32).to(model.device))
    # update the model
    model.online_training(X_tilde)
print("-----online training finished-----")
end = time.time()

# generalizarion error 
# results
results["OnlineAE"] = {}
results["OnlineAE"]["mean"] = np.mean(online_mse)
results["OnlineAE"]["std"] = np.std(online_mse)
results["OnlineAE"]["time"] = end-start


########## OLoKe ##########
# training hyper-parameters
train_args = {
    "d" : d,
    "L" : 1,   # extended dictionary 
    "n_max" : 5, # max prediction horizon
    "off_epochs" : int(5e2),
    "max_iter" : int(5),
    "architecture": [64],
    "lr" : 1e-3,
    "device" : session_device
    }
# Conformal PI hyper-parameters
CP_args = {
    "alpha" : 0.10,
    "eta" : 0.5,
    "Csat" : 20,
    "T_warm_up" : T_warm_up}
start = time.time()
# warm up with avalable data
print("-----warm up started-----")
X_warm_up = traj[:, :T_warm_up, :]
model = Naive(train_args).to(session_device)
model.offline_data(X_warm_up)
model.offline_training(model.off_epochs)
print("-----warm up finished-----")

# online training
online_mse = np.empty(T-1-T_warm_up)
print("-----online training started-----")
for t in range(T_warm_up + 1, T):
    X_tilde = traj[:, t-model.n_max : t+1, :]
    # get online error
    online_mse[t-1-T_warm_up] = model.l_steps_error(torch.tensor(X_tilde, dtype=torch.float32).to(model.device))
    # update the model
    model.online_training(X_tilde, t)
print("-----online training finished-----")
end = time.time()

# generalizarion error 
# results
results["OLoKe"] = {}
results["OLoKe"]["mean"] = np.mean(online_mse)
results["OLoKe"]["std"] = np.std(online_mse)
results["OLoKe"]["time"] = end-start

########## OnlineEDMD ##########

degree = 1

X_train = traj[:,:-1,:]
Y_train = traj[:,1:,:]

# initialization
X_warm_up = X_train[:,:T_warm_up,:].reshape(-1,d).T
Y_warm_up = Y_train[:,:T_warm_up,:].reshape(-1,d).T

start = time.time()
oedmd = OnlineEDMD(d, Phi_poly, T_warm_up)  
oedmd.offline_training(X_warm_up, Y_warm_up, degree)

# online training
online_mse = np.empty(T-1-T_warm_up)
for t in range(T_warm_up, T-1):
    # reconstruction to get online mse
    oedmd.reconstruction()
    # get online mse 
    online_mse[t-T_warm_up] = ((Y_train[0,t:t+1,:].T - oedmd.prediction(X_train[0,t:t+1,:].T)) ** 2).mean().item() 
    # update the model
    oedmd.online_update(X_train[0,t:t+1,:].T, Y_train[0,t:t+1,:].T)
end = time.time()

# storage
results["OEDMD"] = {}
results["OEDMD"]["mean"] = np.mean(online_mse)
results["OEDMD"]["std"] = np.std(online_mse)
results["OEDMD"]["time"] = end-start

print(results)