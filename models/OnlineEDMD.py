import numpy as np
from numpy.linalg import pinv, inv
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
#### dictionarys
def Phi_proj(X):
    """
    X: (d, N)
    """
    def _Phi(x):
        z = x
        return z
    Z = np.array([_Phi(x) for x in X.T])
    return Z.T

def Phi_rbf(X, centers, sigma=0.3):
    """
    X: (d, N)
    """
    def _Phi(x, centers, sigma):        
        diffs = x - centers  # shape (nb_centers, d)
        squared_dist = np.sum(diffs**2, axis=-1)  # shape (nb_centers,)
        return np.exp(-squared_dist / sigma**2)  # shape (nb_centers,)
    Z = np.array([_Phi(x, centers, sigma) for x in X.T])
    return Z.T

def Phi_poly(X, degree):
    """
    X: (d, N) numpy array
    return : (D, N)
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Z = poly.fit_transform(X.T)  
    return Z.T  

#### Online EDMD class
class OnlineEDMD():
    def __init__(self, d, Phi, T_warm_up):
        super().__init__()
        self.Phi = Phi
        #self.centers = None
        self.degree = None
        self.d = d
        self.M = T_warm_up
        
        # snapshots
        self.X_p = None
        self.X_f = None
        
        # K matrix
        self.K_M = None
        self.phi_M = None
        self.phi_M_inv = None
        self.z_M = None
        
        # reconstruction matrix
        self.C = None
        self.lambda_reg=1e-6
    def offline_training(self, X_p, X_f, degree):
        #if n_centers == None:
        #    pass
        #else:
        #    kmeans = KMeans(n_clusters=n_centers, random_state=0)
        #    kmeans.fit(X_p.T)  # fit(X) X (N,d)
        #    # Get centers (shape: 60 x d)
        #    self.centers = kmeans.cluster_centers_
        self.degree = degree
        # stock snapshots
        self.X_p = X_p
        self.X_f = X_f
        # transform by fixed dictionary
        Y_p = self.Phi(self.X_p, self.degree)
        Y_f = self.Phi(self.X_f, self.degree)
        
        # initialize K
        self.phi_M = Y_p @ Y_p.T
        self.phi_M_inv = pinv(self.phi_M)
        self.z_M = Y_f @ Y_p.T 
        self.K_M = self.z_M @ self.phi_M_inv
        
    def online_update(self, x_p_new, x_f_new):
        # stock new snapshot
        self.X_p = np.concatenate((self.X_p, x_p_new), 1)
        self.X_f = np.concatenate((self.X_f, x_f_new), 1)
        self.M += 1
        
        # transform by fixed dictionary
        y_p_new = self.Phi(x_p_new, self.degree)
        y_f_new = self.Phi(x_f_new, self.degree)
           
        # update phi by Matrix Inversion Lemma
        self.phi_M = self.phi_M + y_p_new @ y_p_new.T
        self.phi_M_inv = self.phi_M_inv - (self.phi_M_inv @ y_p_new @ y_p_new.T @ self.phi_M_inv) / (1 + y_p_new.T @ self.phi_M_inv @ y_p_new)

        # update z 
        self.z_M = self.z_M + y_f_new @ y_p_new.T

        # update K
        self.K_M = self.z_M @ self.phi_M_inv
    
    
    def reconstruction(self):
        X = np.concatenate((self.X_p[:,-5:-1], self.X_f[:, -1:]), axis=1)
        Z = self.Phi(X, self.degree)
        ZZt = Z @ Z.T
        XZt = X @ Z.T
        k = ZZt.shape[0]
        self.C = XZt @ np.linalg.inv(ZZt + self.lambda_reg * np.eye(k))
        return self.C
        
    def prediction(self, x_0, n_step = 1):
        z_0 = self.Phi(x_0, self.degree)
        z_n_pred = np.linalg.matrix_power(self.K_M, n_step) @ z_0
        x_n_pred = self.C @ z_n_pred
        return x_n_pred

  


        
        
        
        
        
        
        
        
        
        