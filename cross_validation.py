# -*- coding: utf-8 -*-
"""
Created on Mon Aug 06 11:34:21 2018

@author: mchen
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import itertools
import random

class CSCV:
        
    """
    
    cscv
    
    performing combinatorially symmetric cross-validation
    
    Variables:
    
        M: dataframe, matrix of back-testing return series, T (returns) x N (strategies)
        S: number of submatrices, must be even
        func: performance measurement function, applied on resampled data, axis=0
    
    Methods:
    
        fit: simulation
        plot_logits (num_bins=100, normed=False): hist plot
        plot_IS_OOS: scatter plot
        
    Attributes:

        result: simulation result
        prob_overfit: overfitting probability based on lambda dist

    """        

    def __init__(self, return_matrix, num_submatrices, performance_stats_func):
        
        self.M=return_matrix
        self.S=num_submatrices
        self.func=performance_stats_func
        self.result=pd.DataFrame()
        self.prob_overfit=np.nan
        
        T=self.M.shape[0]
        N=self.M.shape[1]
        
        if (T <= 2 * N):
        
            print ("""warning, Bailey (2015) et al suggest that T (observations) 
                      should be 2x the number of parameters (N), due to the fact
                      that CSCV compares combinations of T/2 trials to their 
                      complements.""")
             
            sys.exit()
             
        if self.S % 2!=0:
             
            print ("warning, S must be even")

            sys.exit()
         
        return
         
    def merge_partitions(self, partitions, list_num_par):
        
        df=pd.DataFrame()
        
        for num_par in list_num_par:
            
            df=df.append(partitions[num_par])
        
        return df
    
    def fit(self):
        
        np.random.seed(32)
        
        partitions=np.array_split(self.M,self.S)
        list_partitions=range(self.S)
        random.shuffle(list_partitions)        
        list_combinations=[rand_list for rand_list in itertools.combinations(list_partitions, self.S/2)]
            
        for rand_list in list_combinations:
                            
            list_train=list(rand_list)
            list_test=(list(set(list_partitions)-set(rand_list)))
            
            random.shuffle(list_train)
            random.shuffle(list_test)                            
                            
            df_train=self.merge_partitions(partitions, list_train)
            df_test=self.merge_partitions(partitions, list_test)
        
            r=df_train.apply(self.func,axis=0)
            
            r_star=r.max()
            
            n_star=r.idxmax()
            
            r_bar=df_test.apply(self.func,axis=0)
            
            r_bar_star=r_bar.ix[n_star]
            
            omegabar=r_bar.rank().ix[n_star]/(len(r_bar)+1)
            
            lambda_=np.log(omegabar / (1 - omegabar))
            
            df_result_temp=pd.DataFrame({'r_star':r_star,'r_bar_star':r_bar_star,'lambda_':lambda_,'omegabar':omegabar},index=[0])
            
            self.result=self.result.append(df_result_temp)
            
        self.prob_overfit=((self.result['lambda_']<0+0).sum()+0.0)/len(self.result)
        
        return 
    
    def plot_logits(self, num_bins=100, normed=False):
        
        plt.figure()
        
        ax=self.result['lambda_'].hist(bins=num_bins, normed=normed)
        
        ax.set_title('Hist. of Rank Logits, '+'Prob Overfit='+str(round(self.prob_overfit,2)))
        
        _=plt.xlabel('Logits')  
        _=plt.ylabel('Frequency')   
        
        plt.show()
        
        return

    def plot_IS_OOS(self):
        
        plt.figure()
        
        _=plt.plot(self.result['r_star'] , self.result['r_bar_star'], marker='.' , linestyle='none')
        
        _=plt.title('OOS Performance Degradation')
        
        _=plt.margins(0.02)
        
        _=plt.xlabel('IS')
        _=plt.ylabel('OOS')
        
        plt.show()
        
        return

class k_fold_CV:
        
    """
        
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

    """        

    def __init__(self, return_matrix, k, performance_stats_func):
        
        self.M=return_matrix
        self.K=k
        self.func=performance_stats_func
        self.result={'train':pd.DataFrame(),
                     'test':pd.DataFrame(),
                     'avg':pd.DataFrame(),}
         
        return
         
    def merge_partitions(self, partitions, list_num_par):
        
        df=pd.DataFrame()
        
        for num_par in list_num_par:
            
            df=df.append(partitions[num_par])
        
        return df
    
    def fit(self):
        
        np.random.seed(32)
        
        partitions=np.array_split(self.M,self.K)
        list_partitions=range(self.K)
        random.shuffle(list_partitions)
        list_combinations=[rand_list for rand_list in itertools.combinations(list_partitions, self.K-1)]
            
        for i, rand_list in enumerate(list_combinations):
                            
            list_train=list(rand_list)
            list_test=(list(set(list_partitions)-set(rand_list)))
            
            random.shuffle(list_train)
            
            df_train=self.merge_partitions(partitions, list_train)
            df_test=self.merge_partitions(partitions, list_test)
        
            r_train=pd.DataFrame(df_train.apply(self.func,axis=0)).transpose()
            r_train.index=[i]
            
            r_test=pd.DataFrame(df_test.apply(self.func,axis=0)).transpose()
            r_test.index=[i]
            
            self.result['train']=self.result['train'].append(r_train)
            self.result['test']=self.result['test'].append(r_test)
       
        df_=pd.DataFrame(self.result['train'].mean(axis=0))
        df_.columns=['avg_IS']
        df_['avg_OOS']=self.result['test'].mean(axis=0)
        df_['avg_IS_OOS']=df_.mean(axis=1)
        
        df_=df_.sort_values(by=['avg_IS_OOS'],ascending=False)   
        
        self.result['avg']=df_

        return 

    def plot_IS_OOS(self):
        
        plt.figure()
        
        _=plt.plot(self.result['train'].stack() , self.result['test'].stack(), marker='.' , linestyle='none')
        
        _=plt.title('IS vs. OOS, k=%s'%self.K)
        
        _=plt.margins(0.02)
        
        _=plt.xlabel('IS')
        _=plt.ylabel('OOS')
        
        plt.show()
        
        return
    
    def plot_avg_IS_OOS(self):
                
        _=self.result['avg'].plot(marker="o")
        
        _=plt.title('avg perf, k=%s'%self.K)
        
        _=plt.margins(0.02)
        
        _=plt.xlabel('strategy')
        _=plt.ylabel('avg perf')
        
        plt.show()

        return        

class CPCV:
        
    """
        
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

    """        

    def __init__(self, return_matrix, N, K, performance_stats_func):
        
        self.M=return_matrix
        self.K=K
        self.N=N        
        self.func=performance_stats_func
        self.result={'train':pd.DataFrame(),
                     'test':pd.DataFrame(),
                     'avg':pd.DataFrame(),}
         
        return
         
    def merge_partitions(self, partitions, list_num_par):
        
        df=pd.DataFrame()
        
        for num_par in list_num_par:
            
            df=df.append(partitions[num_par])
        
        return df
    
    def fit(self):
        
        np.random.seed(32)
        
        partitions=np.array_split(self.M,self.N)
        list_partitions=range(self.N)
        random.shuffle(list_partitions)
        list_combinations=[rand_list for rand_list in itertools.combinations(list_partitions, self.N-self.K)]
            
        for i, rand_list in enumerate(list_combinations):
            
            list_train=list(rand_list)
            list_test=(list(set(list_partitions)-set(rand_list)))
            
            random.shuffle(list_train)
            random.shuffle(list_test)
            
            df_train=self.merge_partitions(partitions, list_train)
            df_test=self.merge_partitions(partitions, list_test)
        
            r_train=pd.DataFrame(df_train.apply(self.func,axis=0)).transpose()
            r_train.index=[i]
            
            r_test=pd.DataFrame(df_test.apply(self.func,axis=0)).transpose()
            r_test.index=[i]
            
            self.result['train']=self.result['train'].append(r_train)
            self.result['test']=self.result['test'].append(r_test)
       
        df_=pd.DataFrame(self.result['train'].mean(axis=0))
        df_.columns=['avg_IS']
        df_['avg_OOS']=self.result['test'].mean(axis=0)
        df_['avg_IS_OOS']=df_.mean(axis=1)
        
        df_=df_.sort_values(by=['avg_IS_OOS'],ascending=False)   
        
        self.result['avg']=df_

        return 

    def plot_IS_OOS(self):
        
        plt.figure()
        
        _=plt.plot(self.result['train'].stack() , self.result['test'].stack(), marker='.' , linestyle='none')
        
        _=plt.title('IS vs. OOS, N=%s, K=%s'%(self.N, self.K))
        
        _=plt.margins(0.02)
        
        _=plt.xlabel('IS')
        _=plt.ylabel('OOS')
        
        plt.show()
        
        return
    
    def plot_avg_IS_OOS(self):
                
        _=self.result['avg'].plot(marker="o")
        
        _=plt.title('avg perf, N=%s, K=%s'%(self.N, self.K))
        
        _=plt.margins(0.02)
        
        _=plt.xlabel('strategy')
        _=plt.ylabel('avg perf')
        
        plt.show()

        return    
        
if __name__=="__main__":
        
    pass
