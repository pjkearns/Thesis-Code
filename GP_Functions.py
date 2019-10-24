import numpy as np

def EXP_cov(x,y,l):
    diff  = np.subtract.outer(x,y)
    return np.exp(-2*(np.sin(diff/2)/l)**2)

def EXP_predict(x_new, x, y, mu_x, mu_y, l, epsilon): # epsilon for noisy
    Kxy = EXP_cov(x_new, x, l) 
    Ky  = EXP_cov(x, x, l) + (epsilon**2)*np.identity(len(x))
    Kx  = EXP_cov(x_new, x_new, l) 

    y_pred = mu_x + Kxy.dot(np.linalg.inv(Ky)).dot(y - mu_y)
    sigma_new = Kx - Kxy.dot(np.linalg.inv(Ky).dot(Kxy.T))
    
    return y_pred, sigma_new

class search_data:
    
    def __init__(self):
        
        self.search_points = []
        self.dists = []
        self.intervals = []
        self.tot_dist = 0

   
    # Expected optimal fractions 
    def GP_search(self, policy, theta, ub, lb, x_init): 
        """For search over 2D-GP boundary with start point"""
        self.err = 0
        self.est = 0
        self.search_points.append(x_init)
        
        for x in policy:

            if self.search_points[-1] < theta:
                lb = self.search_points[-1]
                interval = ub - lb
                self.search_points.append(lb + interval*x)
            else:
                ub = self.search_points[-1]        
                interval = ub - lb
                self.search_points.append(ub - interval*x)

            self.dists.append(abs(self.search_points[-1] - self.search_points[-2]))
            self.intervals.append(interval)
        
        if self.search_points[-1] < theta:
            lb = self.search_points[-1]
        else:
            ub = self.search_points[-1]        
        interval = ub - lb
        self.intervals.append(interval)
        self.tot_dist = np.sum(np.array(self.dists))
        self.est = (ub + lb)/2
        self.err = interval       