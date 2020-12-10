    Package provides a set tools for cross-validation purpose along with one example.		
	
	#######################
    cscv (M, S, func)
	#######################
    
    Combinatorially symmetric cross-validation to estimate prob of backtest overfitting.
    The test is designed to evaluate whether strategy selection (based on a performance
    function) adds value or not. This is different to selecting optimal parameter 
    combination.
    
    The 'cscv' performs the CSCV algorithm as detailed by Bailey et al (2015) 
    The Probability of Backtest Overfitting. Given a true matrix 'M', cscv will 
    (1) split 'M' into 'S' number of sub-matrices, 
    (2) form all sub-matrix combinations taken in groups of size S/2, and 
    (3) perform CSCV given an evaluation function, 'FUN'.
    
    Variables:
    
        M: dataframe, matrix of back-testing return series, T (returns) x N (strategies)
           T > 2*N.    
        S: int, number of submatrices, must be even.
        func: performance measurement function, applied on resampled data, axis=0.
    
    Methods:
    
        fit: simulation
        plot_logits (num_bins=100, normed=False): hist plot
        plot_IS_OOS: scatter plot
        
    Attributes:

        result: df, simulation result
        prob_overfit: float, overfit probability based on lambda dist
		
	
	#######################
	k_fold_CV (M, K, func)
	#######################
	
    performing k-fold cross-validation to select model parameter combination
    
    Variables:
    
        M: dataframe, matrix of back-testing return series, T (returns) x N (strategies)
        K: number of partitions to perform cross-validation
        func: performance measurement function, applied on resampled data, axis=0
    
    Methods:
    
        fit: simulation
        plot_IS_OOS: scatter plot
        plot_avg_IS_OOS: avg performance from simulations
        
    Attributes:

        result: simulation result
		
		
		
	#######################
	CPCV (M, N, K, func)
	#######################
	
    performing combinatorial purged cross-validation to select model parameter combination
    
    Variables:
    
        M: dataframe, matrix of back-testing return series, T (returns) x N (strategies)
        N: number of total submatrices
        K: number of testing submatrices
        func: performance measurement function, applied on resampled data, axis=0
    
    Methods:
    
        fit: simulation
        plot_IS_OOS: scatter plot
        plot_avg_IS_OOS: avg performance from simulations
        
    Attributes:

        result: simulation result		
