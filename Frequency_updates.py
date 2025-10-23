import numpy as np
from models import CKL
from sklearn.model_selection import train_test_split
import torch
import time
import math 
import matplotlib.pyplot as plt
import os
#### using gpu for deep models
session_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(session_device)
print("device: ",session_device)
torch.manual_seed(0)

data_dir = "data/simu_data" 
results = {}

for filename in os.listdir(data_dir):
    filepath = os.path.join(data_dir, filename)
    key = os.path.splitext(filename)[0]  
    trajs = np.load(filepath)
    
    train_idx, test_idx = train_test_split(list(range(trajs.shape[0])), train_size=0.5, random_state=0)
    trajs_train = trajs[train_idx]
    trajs_test = trajs[test_idx]    
    n_traject = trajs_train.shape[0]
    T = trajs_train.shape[1]
    d = trajs_train.shape[2] 
    T_warm_up = int(1/5 * T)
    
    
    ########## COLoKe ##########
    
    results[key] = {}
    train_args = {
        "d": d,
        "L": math.ceil(d/2),
        "n_max": 3,
        "off_epochs": int(2e3),
        "max_iter": int(1e2),
        "lr": 1e-2,
        "architecture": [32, 16],
        "device": session_device
    }
    # Conformal PI hyper-parameters
    CP_args = {
        "alpha": 0.5,
        "eta": 0.1,
        "Csat": 5,
        "T_warm_up": T_warm_up
    }
    
    start = time.time()
    print("-----warm up started-----")
    X_warm_up = trajs_train[:, :CP_args["T_warm_up"], :]
    model = CKL(train_args, CP_args).to(session_device)
    model.offline_data(X_warm_up)
    model.offline_training(model.off_epochs)
    print("-----warm up finished-----")
    
    # Initialize q
    model.initialization(X_warm_up)
    
    # tracking updates
    update_flags = []  # 1 if update triggered, 0 otherwise
    
    
    # online training
    print("-----online training started-----")
    online_mse = np.empty(T-1-T_warm_up)
    for t in range(model.T_warm_up + 1, T):
        X_tilde = trajs_train[:, t-model.n_max : t+1, :]
        # get online error
        online_mse[t-1-T_warm_up] = model.l_steps_error(torch.tensor(X_tilde, dtype=torch.float32).to(model.device))
        # update the model
        updated = model.online_training(X_tilde, t)
        update_flags.append(1 if updated else 0)
        
    print("-----online training finished-----")
    end = time.time()
    
    # stats about updates
    update_count = np.sum(update_flags)
    update_ratio = update_count / len(update_flags)
    avg_interval = (len(update_flags) / update_count) if update_count > 0 else None
    
    print(f"[COLoKe] Updates triggered for {key} dataset: {update_count} ({update_ratio*100:.1f}%)")
    if avg_interval:
        print(f"Average interval between updates: {avg_interval:.2f} steps")
    flags = np.array(update_flags)
    update_indices = np.where(flags == 1)[0]
    update_indices = np.insert(update_indices, 0, 0)
    if len(update_indices) > 1:
        intervals = np.diff(update_indices)   # differences between consecutive updates
        smallest_interval = int(intervals.min())
        biggest_interval = int(intervals.max())
        mean_interval = intervals.mean()
    else:
        smallest_interval = None
        biggest_interval = None
    
    # store update stats in results
    results[key].update({
        "update_flags": update_flags,
        "update_count": int(update_count),
        "update_ratio": float(update_ratio),
        "biggest_interval": float(biggest_interval),
        "smallest_interval" : float(smallest_interval),
        "mean_interval" : mean_interval,
        "avg_interval": float(avg_interval) if avg_interval else None,
        "time": end-start,
        "online error" : online_mse.mean().item(),
        "test error" : model.compute_error(trajs_test)
    })

for key in results.keys():
    if "update_flags" not in results[key]:
        continue  # skip keys without flags

    flags = results[key]["update_flags"]

    plt.figure(figsize=(10, 3))
    plt.plot(flags, drawstyle='steps-post', label=f"Update flags for {key}")
    plt.xlabel("Time step")
    plt.ylabel("Update Flag (0/1)")
    plt.title(f"Update flags timeline - {key}")
    plt.legend()
    plt.grid(True)
    plt.show()

for key in results.keys():
    print(key)
    print("biggest_interval", results[key]["biggest_interval"])
    print("avg_interval", results[key]["avg_interval"])
