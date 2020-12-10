# -*- coding: utf-8 -*-
"""
Created on Tue Aug 07 13:25:51 2018

@author: mchen
"""

import cross_validation
import pandas as pd
import numpy as np

df = (pd.DataFrame(np.random.rand(5000, 1000))-0.5)*2/100
annualised_sharpe=lambda df: df.mean(axis=0)/df.std(axis=0)*(252**0.5)

"""CSCV example"""

cscv=cross_validation.CSCV(df, 10, annualised_sharpe)
cscv.fit() 
cscv.plot_IS_OOS()
cscv.plot_logits()    
    
"""k_fold_CV example"""

k_cv=cross_validation.k_fold_CV(df, 8, annualised_sharpe)
k_cv.fit() 
k_cv.plot_IS_OOS()
k_cv.plot_avg_IS_OOS()

"""CPCV example"""

cpcv=cross_validation.CPCV(df, 16, 2, annualised_sharpe)
cpcv.fit() 
cpcv.plot_IS_OOS()
cpcv.plot_avg_IS_OOS()
    
    
