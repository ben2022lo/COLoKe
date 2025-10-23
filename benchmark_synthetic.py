from models import CKL, Naive, OnlineEDMD, OnlineAE, Phi_poly
from odmd import OnlineDMD
import math
import os
import torch
import numpy as np
from sklearn.model_selection import KFold
import time


#### using gpu for deep models
session_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(session_device)
print("device: ",session_device)
torch.manual_seed(0)

#### load simulated data sets
data_dir = "data/simu_data"  
results = {}
k_fold = 5
for filename in os.listdir(data_dir):
    filepath = os.path.join(data_dir, filename)
    key = os.path.splitext(filename)[0]  # filename without extension
    trajs = np.load(filepath)
    results[key] = {}
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
    
    # initialize per-method storage
    method_names = ["COLoKe", "ODMD", "OEDMD", "OnlineAE", "OLoKe", "One-step"]  
    test_errors = {method: [] for method in method_names}
    online_errors = {method: [] for method in method_names}
    times = {method: [] for method in method_names}
    
    for train_idx, test_idx in kf.split(trajs):
        trajs_train = trajs[train_idx]
        trajs_test = trajs[test_idx]

        trajs_train = trajs[train_idx]
        trajs_test = trajs[test_idx]
        n_traject = trajs_train.shape[0]
        T = trajs_train.shape[1]
        d = trajs_train.shape[2] # dynamic dimension
        T_warm_up = int(1/5 * T)
        
        
        
        ########## OLoKe ##########
        
        #### NaiveDeeplearner
        results[key]["OLoKe"] = {}
        # training hyper-parameters
        train_args = {
            "d" : d,
            "L" : math.ceil(d/2),   # extended dictionary 
            "n_max" : 3, # max prediction horizon
            "off_epochs" : int(2e3),
            "max_iter" : int(100),
            "lr" : 1e-2,
            "architecture": [32, 16],
            "device" : session_device
            }
        
        start = time.time()
        # warm up with avalable data
        print("-----warm up started-----")
        X_warm_up = trajs_train[:, :T_warm_up, :]
        model = Naive(train_args).to(session_device)
        model.offline_data(X_warm_up)
        model.offline_training(model.off_epochs)
        print("-----warm up finished-----")
    
        # online training
        online_mse = np.empty(T-1-T_warm_up)
        print("-----online training started-----")
        for t in range(T_warm_up + 1, T):
            X_tilde = trajs_train[:, t-model.n_max : t+1, :]
            # get online error
            online_mse[t-1-T_warm_up] = model.l_steps_error(torch.tensor(X_tilde, dtype=torch.float32).to(model.device))
            # update the model
            model.online_training(X_tilde, t)
        print("-----online training finished-----")
        end = time.time()
        # evalution on test trajectories
        train_error = model.compute_error(trajs_train)
        test_error = model.compute_error(trajs_test)
        # storage
        test_errors["OLoKe"].append(test_error)
        times["OLoKe"].append(end - start)
        online_errors["OLoKe"].append(online_mse.mean().item())
        
        
        ########## OnlineAE ##########
        
        results[key]["OnlineAE"] = {}
        train_args = {
            "d" : d,
            "L" : d + math.ceil(d/2),   # extended dictionary 
            "n_max" : 3, # max prediction horizon
            "off_epochs" : int(2e3),
            "max_iter" : int(100),
            "lr" : 1e-2,
            "architecture": [32,16],
            "device" : session_device
            }
        start = time.time()
        # warm up with available data
        print("-----warm up started-----")
        X_warm_up = trajs_train[:, :T_warm_up, :]
        model = OnlineAE(train_args).to(session_device)
        model.offline_training(X_warm_up)
        print("-----warm up finished-----")
    
        # online training
        online_mse = np.empty(T-1-T_warm_up)
        print("-----online training started-----")
        for t in range(T_warm_up + 1, T):
            X_tilde = trajs_train[:, t-model.n_max : t+1, :]
            # get online error
            online_mse[t-1-T_warm_up] = model.l_steps_error(torch.tensor(X_tilde, dtype=torch.float32).to(model.device))
            # update the model
            model.online_training(X_tilde)
        print("-----online training finished-----")
        end = time.time()
        # evalution on test trajectories
        train_error = model.compute_error(trajs_train)
        test_error = model.compute_error(trajs_test)
        # storage
        test_errors["OnlineAE"].append(test_error)
        times["OnlineAE"].append(end - start)
        online_errors["OnlineAE"].append(online_mse.mean().item())
        
        
        
        ########## COLoKe ##########
        
        results[key]["COLoKe"] = {}
        # training hyper-parameters
        train_args = {
            "d" : d,
            "L" : math.ceil(d/2),   # extended dictionary 
            "n_max" : 3, # max prediction horizon
            "off_epochs" : int(2e3),
            "max_iter" : int(1e2),
            "lr" : 1e-2,
            "architecture": [32, 16],
            "device" : session_device
            }
        # Conformal PI hyper-parameters
        CP_args = {
            "alpha" : 0.5,
            "eta" : 0.1,
            "Csat" : 5,
            "T_warm_up" : T_warm_up}
        
        start = time.time()
        # warm up with avalable data
        print("-----warm up started-----")
        X_warm_up = trajs_train[:, :CP_args["T_warm_up"], :]
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
            X_tilde = trajs_train[:, t-model.n_max : t+1, :]
            # get online error
            online_mse[t-1-T_warm_up] = model.l_steps_error(torch.tensor(X_tilde, dtype=torch.float32).to(model.device))
            # update the model
            model.online_training(X_tilde, t)
            
        print("-----online training finished-----")
        end = time.time()
        # evalution on test trajectories
        train_error = model.compute_error(trajs_train)
        test_error = model.compute_error(trajs_test)
        # storage
        test_errors["COLoKe"].append(test_error)
        times["COLoKe"].append(end - start)
        online_errors["COLoKe"].append(online_mse.mean().item())
        
        
        
        ########## ODMD ##########
        
        results[key]["ODMD"] = {}
        X_train = trajs_train[:,:-1,:]
        Y_train = trajs_train[:,1:,:]
        #### initialization
        X_warm_up = X_train[:,:T_warm_up,:].reshape(-1,d).T
        Y_warm_up = Y_train[:,:T_warm_up,:].reshape(-1,d).T
        
        start = time.time()
        model = OnlineDMD(d, 1)
        model.initialize(X_warm_up, Y_warm_up)
        
        
        #### online training
        online_mse = np.empty((n_traject, T-1-T_warm_up)) 
        for t in range(T_warm_up, T-1):
            for i in range(n_traject):
                # get online mse 
                online_mse[i,t-T_warm_up] = ((Y_train[i,t,:].T - model.A @ X_train[i,t,:].T) ** 2).mean().item() 
                # update the model
                model.update(X_train[i,t,:].T, Y_train[i,t,:].T)
                
        end = time.time()
        
        #### evaluation
        X_test = trajs_test[:,:-1,:].reshape(-1,d).T
        Y_test = trajs_test[:,1:,:].reshape(-1,d).T
        Y_pred = model.A @ X_test
        test_error = ((Y_pred - Y_test) ** 2).mean().item()

        # storage
        online_errors["ODMD"].append(np.mean(online_mse).item())
        test_errors["ODMD"].append(test_error)
        times["ODMD"].append(end - start)
        
        
        
        ########### OEDMD ##########
        
        #### OnlineEDMD
        n_centers = 60 
        degree = 2 
        results[key]["OEDMD"] = {}
        X_train = trajs_train[:,:-1,:]
        Y_train = trajs_train[:,1:,:]
        
        #### initialization
        X_warm_up = X_train[:,:T_warm_up,:].reshape(-1,d).T
        Y_warm_up = Y_train[:,:T_warm_up,:].reshape(-1,d).T
        
        start = time.time()
        oedmd = OnlineEDMD(d, Phi_poly, T_warm_up)  
        oedmd.offline_training(X_warm_up, Y_warm_up, degree)
    
        #### online training
        online_mse = np.empty((n_traject, T-1-T_warm_up))
        for t in range(T_warm_up, T-1):
            for i in range(n_traject):
                # reconstruction to get online mse
                oedmd.reconstruction()
                # get online mse 
                online_mse[i,t-T_warm_up] = ((Y_train[i,t:t+1,:].T - oedmd.prediction(X_train[i,t:t+1,:].T)) ** 2).mean().item() 
                # update the model
                oedmd.online_update(X_train[i,t:t+1,:].T, Y_train[i,t:t+1,:].T)
        end = time.time()
        

        #### evaluation
        X_test = trajs_test[:,:-1,:].reshape(-1,d).T
        Y_test = trajs_test[:,1:,:].reshape(-1,d).T
        Y_pred = oedmd.prediction(X_test)
        test_error = ((Y_pred - Y_test) ** 2).mean().item()
        
        # storage
        test_errors["OEDMD"].append(test_error)
        times["OEDMD"].append(end - start)
        online_errors["OEDMD"].append(np.mean(online_mse).item())

    for method in method_names:
        results[key][method] = {
            "test error mean": float(np.mean(test_errors[method])),
            "test error std": float(np.std(test_errors[method])/np.sqrt(k_fold)),
            "online error mean": float(np.mean(online_errors[method])),
            "online error std": float(np.std(online_errors[method])/np.sqrt(k_fold)),
            "avg time": float(np.mean(times[method]))
        }



    

        
        
