import numpy as np
import pandas as pd
from models import CKL, Naive
from sklearn.preprocessing import MinMaxScaler
import torch
import time
import math 
import mne
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from pathlib import Path
#### using gpu for deep models
session_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(session_device)
print("device: ",session_device)
torch.manual_seed(0)


# Load EEG data
raw = mne.io.read_raw_edf("data/real_EEG/S001R01.edf", preload=True)
data, times = raw.get_data(return_times=True)
channels, time_steps = data.shape  
# downsampling
new_T = 976  
window_size = time_steps // new_T  # 9760 // 976 = 10
data_subsampled = data.reshape(channels, new_T, window_size).mean(axis=2)
scaler = MinMaxScaler(feature_range=(0, 1))  # scale each channel to [0, 1]
data_scaled = scaler.fit_transform(data_subsampled.T)  # shape = (200, 64)

# Plot normalized trajectories
plt.figure(figsize=(12, 8))
for ch in range(data_scaled.shape[1]):  # loop over 64 channels
    plt.plot(data_scaled[:, ch], label=f"Ch {ch}")  
plt.title("64 EEG channel trajectories (standardized, first 200 samples)")
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
results["COLoKe"] = {
    "mean": np.mean(online_mse),
    "std": np.std(online_mse),
    "time": end-start
}


########## OLoKe (multiple max_iter) ##########

max_iter_list = [1, 5, 10, 50, 100, 150]  # test different max_iter values
results["OLoKe"] = {}

for mi in max_iter_list:
    torch.manual_seed(0)
    print(f"\n====== OLoKe with max_iter = {mi} ======")
    ########## OLoKe ##########
    # training hyper-parameters
    train_args = {
        "d" : d,
        "L" : 1,   # extended dictionary 
        "n_max" : 5, # max prediction horizon
        "off_epochs" : int(5e2),
        "max_iter" : int(mi),
        "architecture": [64],
        "lr" : 1e-3,
        "device" : session_device
        }
    # Conformal PI hyper-parameters
    CP_args = {
        "alpha" : 0.50,
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

    # results
    results["OLoKe"][f"max_iter={mi}"] = {
        "mean": np.mean(online_mse),
        "std": np.std(online_mse),
        "time": end-start
    }
    

rows = []
rows.append({
    "Label": "COLoKe",
    "Mean": float(results["COLoKe"]["mean"]),
    "Std":  float(results["COLoKe"]["std"]),
    "Time": float(results["COLoKe"]["time"]),
})
for variant, stats in results["OLoKe"].items():
    label = f"O-{variant.split('=')[-1]}"
    rows.append({
        "Label": label,
        "Mean": float(stats["mean"]),
        "Std":  float(stats["std"]),
        "Time": float(stats["time"]),
    })

df = pd.DataFrame(rows)

#### Pareto frontier 
def pareto_front_min(df, x_col="Time", y_col="Mean"):
    pts = df[[x_col, y_col]].to_numpy()
    n = len(pts)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        xi, yi = pts[i]
        better_or_equal = (pts[:,0] <= xi) & (pts[:,1] <= yi)
        strictly_better = (pts[:,0] < xi) | (pts[:,1] < yi)
        dominated[i] = np.any(better_or_equal & strictly_better)
    front = df.loc[~dominated].copy()
    front.sort_values([x_col, y_col], inplace=True, ignore_index=True)
    return front

front = pareto_front_min(df)

plt.style.use('figures/utils/tex.mplstyle')  


def set_size(width, fraction=1, subplots=(1,1)):
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    fig_width_in = fig_width_pt * inches_per_pt
    golden_ratio = (math.sqrt(5) - 1) / 2
    fig_height_in = fig_width_in * golden_ratio * (subplots[0]/subplots[1])
    return fig_width_in, fig_height_in

width = 430.00462
fraction = 0.4
s0, s1 = (1, 1)  # single plot
figsize = set_size(width, fraction=fraction, subplots=(s0, s1))
fig = plt.figure(figsize=(figsize[0], figsize[1]))
gs = gridspec.GridSpec(s0, s1)
ax = fig.add_subplot(gs[0, 0])

# Formatter for scientific notation
def sci_fmt(x, _):
    return f"{x:.1e}"
formatter = FuncFormatter(sci_fmt)

# Scatter of all methods
ax.scatter(df["Time"], df["Mean"], marker='o', s=20)
for _, r in df.iterrows():
    if r["Label"] == "COLoKe":
        ax.scatter(r["Time"], r["Mean"], marker='o', s=20, color="red", label="COLoKe")
        ax.annotate(r["Label"], (r["Time"], r["Mean"]), xytext=(4, 0), textcoords="offset points", fontsize=7)
    else:
        ax.scatter(r["Time"], r["Mean"], marker='o', s=20, color="gray")
        if r["Label"] == "O-150":
            ax.annotate(r["Label"], (r["Time"], r["Mean"]), xytext=(-16, 4), textcoords="offset points", fontsize=7)
        elif r["Label"] == "O-100":
            ax.annotate(r["Label"], (r["Time"], r["Mean"]), xytext=(4, -2), textcoords="offset points", fontsize=7)
        else:
            ax.annotate(r["Label"], (r["Time"], r["Mean"]), xytext=(4, -2), textcoords="offset points", fontsize=7)

ax.plot(front["Time"], front["Mean"], linestyle='--', linewidth=10, color="black")
ax.set_xlabel("Execution time (s)")
ax.set_ylabel("Online error")

# scientific notation on y-axis
ax.yaxis.set_major_formatter(formatter)

# Axis style
def style_axis(axis):
    axis.tick_params(length=2, width=0.3, pad=1.1)
    axis.xaxis.set_major_locator(ticker.MaxNLocator(5))
    axis.yaxis.set_major_locator(ticker.MaxNLocator(5))
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.4)
        spine.set_edgecolor('black')

style_axis(ax)
 
# Save the plot
plot_path = Path.cwd().joinpath("figures/pareto_front.pdf").as_posix()
fig.savefig(plot_path, format='pdf', bbox_inches='tight', pad_inches=0.01)
plt.show()
