# %%
from models import CKL
import torch
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import time
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from figures.utils.utils import set_size
from pathlib import Path
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
# using gpu if available
session_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(session_device)
print("device: ",session_device)
torch.manual_seed(0)
# set plot style for figures
plt.style.use('figures/utils/tex.mplstyle') # If you dont have a latex compiler, comment this line
width = 430.00462

# load data
trajs = np.load('data/simu_data/single_attractor.npy')
train_idx, test_idx = train_test_split(list(range(trajs.shape[0])), train_size=0.5, random_state=0)
trajs_train = trajs[train_idx]
trajs_test = trajs[test_idx]

n_traject = trajs_train.shape[0]
T = trajs_train.shape[1]
d = trajs_train.shape[2] # dynamic dimension
T_warm_up = 20


delta_t = 10/100
true_eigenvalues = [-1,-0.05,-0.1]


#### training hyper-parameters
train_args = {
    "L" : 1,   # extended dictionary
    "d" : d,
    "n_max" : 5, # max prediction horizon
    "off_epochs" : int(4e3),
    "max_iter" : 1e3,
    "lr" : 9e-4,
    "device" : session_device
    }
offline_epochs = int(8e3) # for offline model
#### Conformal PI hyper-parameters
CP_args = {
    "alpha" : 0.5,
    "eta" : 0.1,
    "Csat" : 5,
    "T_warm_up" : T_warm_up}

online = True

# %%
#### train 
model = CKL(train_args, CP_args).to(session_device)

if online:
    # Online training
    track_eigenvalues = []
    # Initialize lists for times and test errors
    times = []
    test_losses = []
    
    # warm up with avalable data
    print("-----warm up started-----")
    X_warm_up = trajs_train[:, :CP_args["T_warm_up"], :]
    model.offline_data(X_warm_up)
    
    # Record initial test error and time for online model
    initial_test_error = model.compute_error(trajs_test)
    test_losses.append(initial_test_error)
    times.append(0.0)  # Start at time 0
    
    start_time = time.time()
    total_epochs = 0
    for eval in range(20):
        model.offline_training(model.off_epochs//20)
        total_epochs += model.off_epochs//20
        test_losses.append(model.compute_error(trajs_test))
        times.append(time.time() - start_time)
        print(f"Epoch {total_epochs}/{model.off_epochs}, Loss: {model.losses[-1]}, Test Error: {test_losses[-1]}")
    print("-----warm up finished-----")
    eigenvalues = model.get_eigenvalues(delta_t)
    track_eigenvalues.append(eigenvalues.copy())
    # Initialize q
    model.initialization(X_warm_up)
    for param_group in model.optimizer.param_groups:
        param_group['lr'] = train_args["lr"] * 0.1    

    # Track total iterations for online model
    total_iterations_online = model.off_epochs
    # Record initial test error and time for online model
    initial_test_error = model.compute_error(trajs_test)
    test_losses.append(initial_test_error)
    times.append(time.time() - start_time)

    # online training
    print("-----online training started-----")
    for t in range(CP_args["T_warm_up"] + 1, T):
        X_tilde = trajs_train[:, t-model.n_max : t+1, :]
        model.online_training(X_tilde, t)
        eigenvalues = model.get_eigenvalues(delta_t)
        track_eigenvalues.append(eigenvalues.copy())

        # Update total iterations and record test error with time
        total_iterations_online += model.iter_t[-1]+1
        test_error = model.compute_error(trajs_test)
        test_losses.append(test_error)
        times.append(time.time() - start_time)
        print(f"Epoch {t}/{T-1}, Loss: {model.losses[-1]}, Test Error: {test_error}")
        
    print("-----online training finished-----")
    end_time = time.time()
    # Save times and test errors
    np.save("results_convergence_analysis/online_times.npy", times)
    np.save("results_convergence_analysis/online_test_losses.npy", test_losses)

else:
    # Offline training
    # Initialize lists for times and test errors
    times = []
    test_losses = []
    
    print("\n-----offline training started-----")
    model.offline_data(trajs_train)
    
    # Record initial test error and time for offline model
    initial_offline_error = model.compute_error(trajs_test)
    test_losses.append(initial_offline_error)
    times.append(0.0)  # Start at time 0
    start_time = time.time()

    # Define number of evaluation points to match online model
    total_epochs = 0
    num_eval = T - CP_args["T_warm_up"] -1
    epochs_per_eval = offline_epochs//num_eval
    for i in range(num_eval):
        model.offline_training(epochs_per_eval)
        total_epochs += epochs_per_eval
        test_losses.append(model.compute_error(trajs_test))
        times.append(time.time() - start_time)
        print(f"Epochs: {total_epochs}/{100*num_eval}, Loss: {model.losses[-1]}, Test Error: {test_losses[-1]}")

    print("-----offline training finished-----")
    end_time = time.time()
    # Save times and test errors
    np.save("results_convergence_analysis/offline_times.npy", times)
    np.save("results_convergence_analysis/offline_test_losses.npy", test_losses)
        
total_time = end_time-start_time
print("Total time = " + str(total_time) + " secs")
#### evalution on test trajectories
train_error = model.compute_error(trajs_train)
test_error = model.compute_error(trajs_test)
print("Final train_error: ", train_error)
print("Final test_error: ", test_error)

# %%
# Plot test error versus training time comparison
fraction = 0.55
figsize = set_size(width, fraction=fraction, subplots=(1, 1))
fig, ax = plt.subplots(figsize=(figsize[0]*0.7,figsize[1]))

# load times and test errors
offline_times = np.load("results_convergence_analysis/offline_times.npy")
offline_test_losses = np.load("results_convergence_analysis/offline_test_losses.npy")
online_times = np.load("results_convergence_analysis/online_times.npy")
online_test_losses = np.load("results_convergence_analysis/online_test_losses.npy")

ax.plot(offline_times, offline_test_losses, label='Offline', color='tab:grey', linewidth=1)
ax.plot(online_times, online_test_losses, label='COLoKe', color='tab:orange', linewidth=1)
ax.set_xlabel(r'train time (seconds)')
ax.xaxis.set_label_coords(0.5, -0.12)  # (x, y) in axis coordinates
ax.set_yscale('log')
# ax.set_ylim(1e-6,1e0)
ax.set_yticks([1e0, 1e-6])  # Customize based on your data
ax.tick_params(axis='y', which='minor', labelsize=10)

ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.03, 1.05), handlelength=0.8)
ax.set_ylabel(r"test error")
ax.yaxis.set_label_coords(-0.12, 0.5)

ax.tick_params(length=2, width=0.4, pad=1.1)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.4)
    spine.set_edgecolor('black')

plt.tight_layout()
plot_path = Path.cwd().joinpath("figures/time_comp.pdf").as_posix()
fig.savefig(plot_path, format='pdf', bbox_inches='tight', pad_inches=0.025)
plt.show()

# %%
#### eigenvalues convergence ####
if online:
    errors = np.abs((true_eigenvalues - np.array(track_eigenvalues))) #the order should be checked
    fraction = 0.4
    s0,s1 = (3,1)
    figsize = set_size(width, fraction=fraction, subplots=(s0, s1))
    fig, ax = plt.subplots(s0, s1, figsize=(figsize[0],figsize[1]*0.6))

    ax[0].plot(range(errors.shape[0]), errors[:, 0], label=r"$|\lambda_1 - \lambda_1^*$", color='tab:orange', linewidth=1)
    ax[0].set_xticks([])
    ax[1].plot(range(errors.shape[0]), errors[:, 1], label=r"$|\lambda_2 - \lambda_2^*$", color='tab:orange', linewidth=1)
    ax[1].set_xticks([])
    ax[2].plot(range(errors.shape[0]), errors[:, 2], label=r"$|\lambda_3 - \lambda_3^*$", color='tab:orange', linewidth=1)
    #ax[2].set_xlabel(r"$t$")
    # ax[0].set_xlabel("training step")
    # ax[0].set_ylabel("abosolute error")
    ax[0].set_yscale("log")  # useful if errors decay exponentially
    ax[1].set_yscale("log")
    ax[2].set_yscale("log")
    ax[2].set_ylim(1.45e-2,1.5e-1)
    ax[2].set_yticks([1e-1])

    ax[0].legend(frameon=False,loc='upper right', handles=[Line2D([0], [0], color='none', label =r"$|\lambda_1 - \lambda_1^*|$")])
    ax[1].legend(frameon=False,loc='upper right', handles=[Line2D([0], [0], color='none', label =r"$|\lambda_2 - \lambda_2^*|$")])
    ax[2].legend(frameon=False,loc='upper right', handles=[Line2D([0], [0], color='none', label =r"$|\lambda_3 - \lambda_3^*|$")])
    for a in ax.ravel():
        a.tick_params(length=2, width=0.5, pad=1.1)  # Smaller ticks and tighter to axis

    plt.subplots_adjust(
    wspace=0.001,
    hspace=0.10,
    left=0.01,
    right=0.99,
    top=0.99,
    bottom=0.01
    )
    #plt.tight_layout()
    plt.show()
    plot_path = Path.cwd().joinpath("figures/eigenvalues.pdf").as_posix()
    fig.savefig(plot_path, format='pdf', bbox_inches='tight', pad_inches=0.01)


# %%
#### eigenfunctions convergence ####
if online:
    fraction = 0.6
    s0,s1 = (2,3)
    figsize = set_size(width, fraction=fraction, subplots=(s0, s1))
    fig, ax = plt.subplots(s0, s1, figsize=(figsize[0]*0.7,figsize[1]))

    # compare estimated eigenfunctions to true eigenfunctions
    dis = 200
    X1, X2 = np.meshgrid(np.linspace(-2, 2, dis), np.linspace(-2, 2, dis))
    domain_points = np.column_stack([X1.ravel(), X2.ravel()])

    # true eigenfunctions
    lambda_ = -1; mu = -0.05
    b = lambda_ / (lambda_ - 2 * mu)
    phi1 =  X2 - b * X1**2 # for mu = -0.05
    phi2 = X1
    phi3 = X1**2

    im1 = ax[0,0].imshow(phi1, extent=[-2, 2, -2, 2], origin='lower', cmap='plasma')
    # ax[0,0].set_ylabel(r"$x_2$")
    ax[0,0].set_yticks([-1.8,0, 1.8],labels=["-2","0","2"])
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([-1.99,0.0001, 1.99], minor=True)  # Add the actual border ticks as minor ticks
    ax[0,0].tick_params(axis='y', which='minor', length=3, width=0.6)
    im2 = ax[0,1].imshow(phi2, extent=[-2, 2, -2, 2], origin='lower', cmap='plasma')
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])

    im3 = ax[0,2].imshow(phi3, extent=[-2, 2, -2, 2], origin='lower', cmap='plasma')
    ax[0,2].set_xticks([])
    ax[0,2].set_yticks([])
    # estimated eigenfunctions
    Z = model.eigen_functions(torch.tensor(domain_points, dtype=torch.float32).to(session_device))

    im1_estim = ax[1,0].imshow(Z[0].reshape((dis, dis)), extent=[-2, 2, -2, 2], origin='lower', cmap='plasma')
    # ax[1,0].set_xlabel(r"$x_1$")
    # ax[1,0].set_ylabel(r"$x_2$")
    ax[1,0].set_yticks([-1.8,0, 1.8],labels=["-2","0","2"])
    ax[1,0].set_xticks([-1.8,0, 1.8],labels=["-2","0","2"])
    ax[1,0].set_yticks([-1.99, 0.0001, 1.99], minor=True)  # Add the actual border ticks as minor ticks
    ax[1,0].tick_params(axis='y', which='minor', length=3, width=0.6)
    ax[1,0].set_xticks([-1.99, 0.0001, 1.99], minor=True)  # Add the actual border ticks as minor ticks
    ax[1,0].tick_params(axis='x', which='minor', length=3, width=0.6)


    im2_estim = ax[1,1].imshow(-Z[1].reshape((dis, dis)), extent=[-2, 2, -2, 2], origin='lower', cmap='plasma')
    # ax[1,1].set_xlabel(r"$x_1$")
    ax[1,1].set_yticks([])
    ax[1,1].set_xticks([-1.8,0, 1.8],labels=["-2","0","2"])
    ax[1,1].set_xticks([-1.99,0.0001, 1.99], minor=True)  # Add the actual border ticks as minor ticks
    ax[1,1].tick_params(axis='x', which='minor', length=3, width=0.6)


    im3_estim = ax[1,2].imshow(-Z[2].reshape((dis, dis)), extent=[-2, 2, -2, 2], origin='lower', cmap='plasma')
    # ax[1,2].set_xlabel(r"$x_1$")
    ax[1,2].set_yticks([])
    ax[1,2].set_xticks([-1.8,0, 1.8],labels=["-2","0","2"])
    ax[1,2].set_xticks([-1.99,0.0001, 1.99], minor=True)  # Add the actual border ticks as minor ticks
    ax[1,2].tick_params(axis='x', which='minor', length=3, width=0.6)

    for a in ax.ravel():
        a.tick_params(length=4, width=0, pad=0.1)  # Smaller ticks and tighter to axis
        for spine in a.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor('black')

            
    plt.subplots_adjust(
    wspace=0.08,
    hspace=0.05,
    left=0.0,
    right=0.99,
    top=0.99,
    bottom=0.00
    )
    plt.show()
    plot_path = Path.cwd().joinpath("figures/eigenfunctions.pdf").as_posix()
    fig.savefig(plot_path, format='pdf', bbox_inches='tight', pad_inches=0.02)


# %%
#### q_t convergence ####
if online:
    fraction = 0.9
    s0,s1 = (1,2)
    figsize = set_size(width, fraction=fraction, subplots=(s0, s1))
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    gs = gridspec.GridSpec(s0, s1, width_ratios=[5, 2.5], wspace=0.2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax = [ax0, ax1]

    ax[0].fill_between(range(20,100), model.qs[:-1], alpha=0.15,facecolor='tab:blue',edgecolor='none',zorder=1)
    ax[0].plot(range(20,100),model.qs[0:-1], label=r"$q_t$", color='tab:blue', linewidth=0.7,zorder=1)
    ax[0].fill_between(range(20,100), np.clip(model.scores_after,0,None), np.array(model.scores[-80:]), color='tab:orange', alpha=0.9,edgecolor='none',zorder=5)
    ax[0].plot(range(20,35), model.scores_after[0:15], label=r"$s_t^{\prime}$", color='tab:orange', linewidth=0.7, linestyle='-',zorder=5)

    ax[0].set_xlabel(r"$t$")
    ax[0].set_yscale("log", nonpositive='clip',subs=[1,2,3,4,5,6,7,8,9,10])
    ax[0].legend(frameon=False, handlelength=1, loc="upper left", bbox_to_anchor=(0.25, 1.05), ncol=2, columnspacing=1, fontsize=11)
    ax[0].set_xlim(20,99)
    ax[0].set_ylim(5e-5,4e-3)

    ax_twinx = ax[0].twinx()
    ax_twinx.scatter(range(20,100), model.iter_t, label=r"$n_{iter}$", color='tab:green', s=0.4,marker='+',linewidth=2)
    ax_twinx.legend(frameon=False, handlelength=1, loc="upper left", bbox_to_anchor=(0.65, 1.05), fontsize=11)
    ax_twinx.set_ylim(-10,900)
    ax_twinx.tick_params(axis='y', colors='tab:green')
    ax_twinx.yaxis.label.set_color('tab:green')

    ax[1].plot(range(20,100), np.cumsum(model.qs[-80:]), label=r"$\displaystyle \sum_{i=t_0}^t q_i$", color='tab:red', linewidth=1)
    ax[1].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax[1].set_xlabel(r"$t$")
    ax[1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.6), handlelength=1, fontsize=11)
  
    def style_axis(axis):
        axis.tick_params(length=2, width=0.4, pad=1.1)
        axis.xaxis.set_major_locator(ticker.MaxNLocator(3))
        axis.yaxis.set_major_locator(ticker.MaxNLocator(3))
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.4)
            spine.set_edgecolor('black')

    style_axis(ax[0])
    style_axis(ax[1])
    style_axis(ax_twinx)

    plot_path = Path.cwd().joinpath("figures/q_t.pdf").as_posix()
    fig.savefig(plot_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.show()


# %%
#### training efficiency ####
if online:
    fraction = 0.5
    s0,s1 = (1,1)
    fig, ax = plt.subplots(s0, s1, figsize=set_size(width, fraction=fraction, subplots=(s0, s1)))

    ax.hist(model.iter_t, bins=20, density=True, alpha=0.7, color='tab:blue', edgecolor='white', linewidth=0.5)
    #ax[0].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    #ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax.tick_params(length=2, width=0.5, pad=1.1)  # Smaller ticks and tighter to axis

    plt.subplots_adjust(
    wspace=0.20,
    hspace=0.10,
    left=0.01,
    right=0.99,
    top=0.99,
    bottom=0.01
    )
    plt.show()
    plot_path = Path.cwd().joinpath("figures/training_efficiency.pdf").as_posix()
    fig.savefig(plot_path, format='pdf', bbox_inches='tight', pad_inches=0.01)
# else:
#     fraction = 0.3
#     s0,s1 = (1,1)  
#     figsize = set_size(width, fraction=fraction, subplots=(s0, s1))
#     fig, ax = plt.subplots(s0, s1, figsize=(figsize[0],figsize[1]*0.9))

#     x_old = np.linspace(0, 1, 75)
#     x_new = np.linspace(0, 1, 142)
#     # Interpolation function
#     f = interp1d(x_old, model.test_error_evolution, kind='linear')  # or 'cubic', 'quadratic'
#     resampled = f(x_new)
    
#     ax.plot(resampled,linewidth=0.8, label = "offline")
#     ax.hlines(y=1.745e-4, xmin = 0, xmax = 142, color = 'red', linestyle='--', linewidth=0.8, label='online')
#     ax.set_yscale("log") 
#     ax.set_xlabel("second")
#     ax.legend(frameon=False, loc="upper right", handlelength=1, bbox_to_anchor=(1, 1.1))
    
#     ax.tick_params(length=2, width=0.7, pad=1.1)
#     ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
#     #ax.yaxis.set_major_locator(ticker.MaxNLocator(2))
        
#     for spine in ax.spines.values():
#         spine.set_visible(True)
#         spine.set_linewidth(0.5)
#         spine.set_edgecolor('black')
    
#     plt.show()
#     plot_path = Path.cwd().joinpath("figures/training_efficiency.pdf").as_posix()
#     fig.savefig(plot_path, format='pdf', bbox_inches='tight', pad_inches=0.02)