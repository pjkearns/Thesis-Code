import numpy as np

def range_act(thtRange, Xc, Xo, act, lam, policy, S):
    cost = 0
    for theta in thtRange:
    # perform the first step then follow greedy policy
        search = search_data()
        search.Non_uni_search_dist(Xc, Xo, act, theta, lam, policy, S)
        cost += search.cost/len(thtRange)
    return cost

class search_data:
    
    def __init__(self):
        self.dist = 0
        self.cost = 0        
    
    def Non_uni_search_dist(self, Xc, Xo, act, theta, lam, policy, S): 
        
        lb = min(Xc, Xo)
        ub = max(Xc, Xo)
        
        policy = np.insert(policy, 0, act)
        for x in policy:
                
            state = np.abs(ub - lb)
            
            if state > S[1]:
            # Action is the distance into interval
                dist = state*x
                dist = S[np.argmin(np.abs(S - dist))]
                
                if Xc < Xo:   
                    Xc += dist

                elif Xc > Xo:
                    Xc -= dist
                    
                if Xc < theta:
                    lb = Xc
                    Xo = ub
                else:
                    ub = Xc         
                    Xo = lb
                
                self.dist += dist

            else:
                break
        self.cost = lam*self.dist + np.abs(ub - lb)