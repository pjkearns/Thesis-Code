import numpy as np
from scipy.stats import truncnorm

def ent(x): 
    """Returns expected interval size"""
    return np.prod(x**2 + (1-x)**2)

def dst(x): 
    """Returns expected distance"""
    dst = x[0]
    N = len(x)
    for ii in range(1,N):
        dst = dst + x[ii]*np.prod(x[:ii]**2 + (1-x[:ii])**2)    
    return dst   

def J(lam,x):
    """Returns expected cost"""
    return ent(x) + lam*dst(x)

def opt_fracs(N,lam):  
    """Returns N optimal fractions"""
    x = np.array([1/2 - lam/4])
    for ii in range(N-1):        
        x_n = 1/2 - lam/(4*J(lam,x))
        x = np.insert(x,0,x_n)     
    return x

def opt_fracs_rho(N,lam):  
    """Variant of opt_fracs, returns rho as well"""
    x = np.array([1/2 - lam/4])
    rhos = np.array([])
    for ii in range(N-1): 
        rho = J(lam,x)
        x_n = 1/2 - lam/(4*rho)
        rhos = np.insert(rhos,0,rho)
        x = np.insert(x,0,x_n)
    rhos = np.append(rhos, 1)
    return x, rhos

def N_o_f(err,lam):
    """Calculates N expected optimal fractions to certain error size.
    Returns policy, N steps, and expected distance"""
    x = np.array([1/2 - lam/4])
    while np.prod(x**2 + (1-x)**2) > err:
        x_n = 1/2 - lam/(4*J(lam,x))
        x = np.insert(x,0,x_n)      
    dist = dst(x)        
    return x, len(x), dist

def min_time(err,Ts,Tt):
    """Calculate the lambda value and policy to minimize total search
    time to error"""
    min_time = np.inf
    min_lam = np.inf
    
    lams = np.linspace(0.01,1.5,150)
    for lam in lams:
        x, n, d = N_o_f(err,lam)
        time = Ts*n + Tt*d
        if time < min_time:
            min_time = time
            min_lam = lam
            min_x = x
            min_n = n
            min_d = d
    
    return min_lam, min_x

def get_truncated_normal(mean, sd, low, upp):
    """Return truncated normal distribution"""
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


### FUNCTIONS FOR PERFORMING SEARCH ###
class search_data:
    
    def __init__(self):
        
        self.search_points = [0]
        self.dists = []
        self.intervals = []
        self.tot_dist = 0
        self.cost = 0
        self.acts = []
        
    def sim_uni_search(self, lam, theta, policy):
        """Only used for generating continuous simulated results for theoretical comparison"""
        lb = 0
        ub = 1
        interval = ub - lb

        for x in policy:
            if self.search_points[-1] < theta:
                self.search_points.append(lb + interval*x)
            else:
                self.search_points.append(ub - interval*x)

            if self.search_points[-1] < theta:
                lb = self.search_points[-1]
            else:
                ub = self.search_points[-1]        
            
            interval = ub - lb            

            self.dists.append(abs(self.search_points[-2] - self.search_points[-1]))
            self.intervals.append(interval)

        self.tot_dist = np.sum(np.array(self.dists))
        self.cost = lam*self.tot_dist + self.intervals[-1]
        
    def uni_search(self, lam, theta, delta, S, policy): 
        """Used only for discrete, fixed policy from FH or QS"""
        lb = 0
        ub = 1
        interval = ub - lb

        for x in policy:
            if interval > delta:
                if self.search_points[-1] < theta:
                    lb = self.search_points[-1]
                    interval = ub - lb
                    x_ind = np.argmin(np.abs(lb + interval*x - S))
                    self.search_points.append(S[x_ind])
                else:
                    ub = self.search_points[-1]        
                    interval = ub - lb
                    x_ind = np.argmin(np.abs(ub - interval*x - S))
                    self.search_points.append(S[x_ind])                    

                self.dists.append(abs(self.search_points[-2] - self.search_points[-1]))
                self.intervals.append(interval)
            else:
                break

        if interval > delta:
            if self.search_points[-1] < theta:
                lb = self.search_points[-1]
            else:
                ub = self.search_points[-1]        
            interval = ub - lb
        if interval < delta:
            interval = delta
        self.intervals.append(interval)
        self.tot_dist = np.sum(np.array(self.dists))
        self.cost = lam*self.tot_dist + self.intervals[-1]
        
    def uni_search_dist(self, lam, theta, delta, S, policy): 
        """Used for all other uniform policies"""
        lb = 0
        ub = 1
        interval = ub - lb

        for row in policy:
            idL = np.argmin(np.abs(S - interval))
            x = row[idL]
            self.acts.append(x)
            if interval > delta:
                if self.search_points[-1] < theta:
                    lb = self.search_points[-1]
                    interval = ub - lb
                    x_ind = np.argmin(np.abs(lb + interval*x - S))
                    self.search_points.append(S[x_ind])
                else:
                    ub = self.search_points[-1]        
                    interval = ub - lb
                    x_ind = np.argmin(np.abs(ub - interval*x - S))
                    self.search_points.append(S[x_ind])                    

                self.dists.append(abs(self.search_points[-2] - self.search_points[-1]))
                self.intervals.append(interval)
            else:
                break

        if interval > delta:
            if self.search_points[-1] < theta:
                lb = self.search_points[-1]
            else:
                ub = self.search_points[-1]        
            interval = ub - lb
        if interval < delta:
            interval = delta
        self.intervals.append(interval)
        self.tot_dist = np.sum(np.array(self.dists))
        self.cost = lam*self.tot_dist + self.intervals[-1]
        
       
    def Non_uni_search_dist(self, lam, theta, S, policy): 
        """Use for searching Nonuniform distribution by distance"""
        lb = 0
        ub = 1
        Xc = 0
        Xo = 1
        
        N = len(policy)
        
        for ii in range(N):
            idXc = np.argmin(np.abs(Xc-S))
            idXo = np.argmin(np.abs(Xo-S)) 
            
            x = policy[ii,idXc,idXo]
            self.acts.append(x)
                
            state = np.abs(ub - lb)
            
            if np.abs(idXo - idXc) > 1:
            # Action is the distance into interval
                dist = state*x
            
                if Xc < Xo:   
                    newXc = Xc + dist

                elif Xc > Xo:
                    newXc = Xc - dist

                newXc = S[np.argmin(np.abs(S - newXc))]
                dist = np.abs(newXc - Xc)
                
                Xc = newXc
                if Xc < theta:
                    lb = Xc
                    Xo = ub
                else:
                    ub = Xc         
                    Xo = lb
                
                self.tot_dist += dist
                self.search_points.append(Xc)

                idXc = np.argmin(np.abs(Xc-S))
                idXo = np.argmin(np.abs(Xo-S))   

            else:
                break
        while len(self.acts)<N:
            self.acts.append(0)
        self.cost = lam*self.tot_dist + np.abs(ub - lb)
        
    def Non_uni_search_entropy(self, lam, theta, S, policy): 
        """Use for searching by fractions of entropy distribution"""
        lb = 0
        ub = 1
        Xc = 0
        idXc = np.argmin(np.abs(Xc-S))
        Xo = 1
        idXo = np.argmin(np.abs(Xo-S))
        interval = np.abs(Xc-Xo)
        
        N = len(policy)
        
        for ii in range(N):
            if np.abs(idXo - idXc) > 1:
            # Action is the fraction into the distribution travelled
            # policy says what fraction into remaining normal dist to travel
                if Xo > Xc:   
                    tn = get_truncated_normal(mean, std, Xc, Xo)
                    idZ = np.argmin(np.abs(tn.cdf(S) - policy[ii,idXc,idXo]))
                    Z = S[idZ]

                elif Xo < Xc:
                    tn = get_truncated_normal(mean, std, Xo, Xc)
                    idZ = np.argmin(np.abs(tn.cdf(S) - (1-policy[ii,idXc,idXo])))
                    Z = S[idZ]
                
                step = np.abs(Z - self.search_points[-1])
                self.acts.append(step/interval)
                self.search_points.append(Z)
                Xc = self.search_points[-1]

                if Xc < theta:
                    lb = self.search_points[-1]
                    Xo = ub
                else:
                    ub = self.search_points[-1]         
                    Xo = lb

                idXc = np.argmin(np.abs(Xc-S))
                idXo = np.argmin(np.abs(Xo-S))   
                interval = ub - lb
                self.dists.append(step)
                self.intervals.append(interval)
            else:
                break
        while len(self.acts)<N:
            self.acts.append(0)
        final_dist = get_truncated_normal(mean, std, lb, ub)  
        V_f = np.exp(final_dist.entropy())
        self.tot_dist = np.sum(np.array(self.dists))
        self.cost = lam*self.tot_dist + interval
