The purpose of this folder is to provide the critical framework for all of the work described in my thesis.


FINITE HORIZON (CLOSED FORM) SEARCH
***********************************
- FH_Functions.py:
	- Functions to compute closed-form solutions, optimal fractions, cost, etc.
	- Functions to perform search procedures for simulations.
	- Functions to calculate optimal lambda value for Time-constrained search.
	- Functions to generate truncated normal distribution.

- FH Policy Figures:
	- Plot optimal policies for various values of lambda.
	- Compare theoretical and simulated search performances.
	- Compare cost of search against QS methods.
	- Theoretical plot of cost if we penalize # of samples.



2D GAUSSIAN PROCESS SEARCH
***********************************
- GP_Functions.py:
	- Functions to compute covariance kernels and predictions.
	- Functions to perform transect searches.

- GP_search_compare:
	- Determine the optimal #strips vs #samples tradeoff for fixed samples.
	- Compare/plot performance of each combination averaged over 100 boundaries.
	- Compare/plot FH vs Quantile Search performance.

- GP_boundary_plots:
	- Generate/plot progressive, strip-by-strip 2D-FH search.
	- Plot final 2D-FH boundary estimate.
	- Can change # strips, boundary parameters, etc.




REINFORCEMENT LEARNING METHODS
*********************************** 
A brief description of the contents of these files (every file listed has both _Uniform and _Nonuniform versions for the corresponding change point distributions):
    
- DynamicProgramming:
	- Calculates and saves a value table and bestAction policy. 
	- Prints compute time.
    
- QLearning:
	- Calculates and saves a value table and bestAction policy. 
	- Prints compute time.
        
- Rollout:
	- Performs rollout search w/ or w/out policy improvement for range of change points. 
	- Saves the updated heuristic "policy", which is just an average of actions taken.  
	- Prints compute time, average performance.
        
- DQN:
	- Trains a relatively shallow network to learn search procedure. 
	- Saves network parameters and "policy", or average of actions taken. 
	- Prints compute time, average performance.
        
- MethodCompare:
	- Takes policies from DynamicProgramming, QLearning, calculates average performance.
	- Plotting of policies from all methods.

- ALL POLICIES STORED IN THE "Policies" FOLDER





