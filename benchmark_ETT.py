import pandas as pd
from models import CKL, Naive, OnlineEDMD, OnlineAE, Phi_poly
from odmd import OnlineDMD
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import time
#### using gpu for deep models
session_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(session_device)
print("device: ",session_device)
torch.manual_seed(0)

# Load the dataset
df = pd.read_csv('data/real_ETT/ETTh1.csv')

# Extract features and target
data = df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']]
target = df['OT']

# Normalize the features
scaler = StandardScaler()
data = scaler.fit_transform(data)
data = np.expand_dims(data, axis=0)
traj = data[:,:200,:]
traj_test = data[:,200:250,:]
traj_test_pre = traj_test[:,:-1,:]
traj_test_after = traj_test[:,1:,:]
T = traj.shape[1]
d = traj.shape[2]
T_warm_up = int(100)


results = {}

########## OLoKe ##########
# training hyper-parameters
train_args = {
    "d" : d,
    "L" : int(d/2),   # extended dictionary 
    "n_max" : 3, # max prediction horizon
    "off_epochs" : int(2e3),
    "max_iter" : int(1e2),
    "lr" : 1e-2,
    "architecture": [32,16],
    "device" : session_device
    }
# Conformal PI hyper-parameters
CP_args = {
    "alpha" : 0.5,
    "eta" : 0.1,
    "Csat" : 10,
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
test_error = model.compute_error(traj_test)
X_test = torch.tensor(traj_test[:,:-1,:].reshape(-1,d), dtype=torch.float32).to(session_device)
Y_test = torch.tensor(traj_test[:,1:,:].reshape(-1,d), dtype=torch.float32).to(session_device)
Y_pred = model.forward(X_test, 1)[:,:d]
err = (Y_pred - Y_test) ** 2
err = err.cpu().detach().numpy()
err = np.mean(err, axis = 1)
test_error_std = np.std(err)
# results
results["OLeKe"] = {}
results["OLeKe"]["mean"] = np.mean(online_mse)
results["OLeKe"]["std"] = np.std(online_mse)
results["OLeKe"]["time"] = end-start
results["OLeKe"]["test_error"] = np.mean(test_error)
results["OLeKe"]["test_error_std"] = np.mean(test_error_std)


########## COLoKe ##########
# training hyper-parameters
train_args = {
    "d" : d,
    "L" : int(d/2),   # extended dictionary 
    "n_max" : 3, # max prediction horizon
    "off_epochs" : int(2e3),
    "max_iter" : int(1e2),
    "lr" : 1e-2,
    "architecture": [32,16],
    "device" : session_device
    }
# Conformal PI hyper-parameters
CP_args = {
    "alpha" : 0.5,
    "eta" : 0.1,
    "Csat" : 10,
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


# generalizarion error 
test_error = model.compute_error(traj_test)
X_test = torch.tensor(traj_test[:,:-1,:].reshape(-1,d), dtype=torch.float32).to(session_device)
Y_test = torch.tensor(traj_test[:,1:,:].reshape(-1,d), dtype=torch.float32).to(session_device)
Y_pred = model.forward(X_test, 1)[:,:d]
err = (Y_pred - Y_test) ** 2
err = err.cpu().detach().numpy()
err = np.mean(err, axis = 1)
test_error_std = np.std(err)
# results
results["COLeKe"] = {}
results["COLeKe"]["mean"] = np.mean(online_mse)
results["COLeKe"]["std"] = np.std(online_mse)
results["COLeKe"]["time"] = end-start
results["COLeKe"]["test_error"] = np.mean(test_error)
results["COLeKe"]["test_error_std"] = np.mean(test_error_std)




########## OnlineAE ##########
train_args = {
    "d" : d,
    "L" : d + int(d/2),   # extended dictionary 
    "n_max" : 3, # max prediction horizon
    "off_epochs" : int(2e3),
    "max_iter" : int(1e2),
    "lr" : 1e-2,
    "architecture": [32,16],
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
test_error = model.compute_error(traj_test)
X_test = torch.tensor(traj_test[:,:-1,:].reshape(-1,d), dtype=torch.float32).to(session_device)
Y_test = torch.tensor(traj_test[:,1:,:].reshape(-1,d), dtype=torch.float32).to(session_device)
_,_,Y_pred = model.forward(X_test, 1)
err = (Y_pred - Y_test) ** 2
err = err.cpu().detach().numpy()
err = np.mean(err, axis = 1)
test_error = np.mean(err)
test_error_std = np.std(err)

# results
results["OnlineAE"] = {}
results["OnlineAE"]["mean"] = np.mean(online_mse)
results["OnlineAE"]["std"] = np.std(online_mse)
results["OnlineAE"]["time"] = end-start
results["OnlineAE"]["test_error"] = np.mean(test_error)
results["OnlineAE"]["test_error_std"] = np.mean(test_error_std)



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

#evaluation
X_test = traj_test[:,:-1,:].reshape(-1,d).T
Y_test = traj_test[:,1:,:].reshape(-1,d).T
Y_pred = model.A @ X_test
err = (Y_pred - Y_test) ** 2
test_error = np.mean(err)
test_error_std = np.std(err)
# storage
results["ODMD"] = {}
results["ODMD"]["mean"] = np.mean(online_mse)
results["ODMD"]["std"] = np.std(online_mse)
results["ODMD"]["time"] = end-start
results["ODMD"]["test_error"] = np.mean(test_error)
results["ODMD"]["test_error_std"] = np.std(online_mse)


########## OnlineEDMD ##########

degree = 2

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


# evaluation
X_test = traj_test[:,:-1,:].reshape(-1,d).T
Y_test = traj_test[:,1:,:].reshape(-1,d).T
Y_pred = oedmd.prediction(X_test)
err = (Y_pred - Y_test) ** 2
test_error = np.mean(err)
test_error_std = np.std(err)
# storage
results["OEDMD"] = {}
results["OEDMD"]["mean"] = np.mean(online_mse)
results["OEDMD"]["std"] = np.std(online_mse)
results["OEDMD"]["time"] = end-start
results["OEDMD"]["test_error"] = np.mean(test_error)
results["OEDMD"]["test_error_std"] = np.std(online_mse)