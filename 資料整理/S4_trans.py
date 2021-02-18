# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 13:58:37 2020

1 2 4 5 6 7 8 9 10 12 13 14 15 16 17 19 20 22 23 24(16)   3 21(15)    11(14)   18(18)

h3n2
1 2 3 5 6 7 9 10 11 12 14 15 16 (16)   4 8 17(15)   13(14)

sars
1 3(11)    2(10)   4(5)    5 6(9)   7 8(8)   9(2)   10 11(1)


h1n1
1 4 5 14 15 16 19 22 23 24(176) 2 6 7 8 9 10 12 13 17 20(192)  3 21(180) 11(154) 18(198) 

h3n2
1 5 6 7 10 12 15(128) 2 3 9 11 13 14 16(112) 4 17(105) 8(120)

sars
1 3(275) 2(270) 4(155) 5 6(324) 7 8(288) 9(72) 10 11(36)

@author: a1033
"""

import numpy as np 
import pandas as pd
import csv

all_df = pd.read_csv("C:/Users/a1033/OneDrive/桌面/專題/new-test_mlp/MLP_H1N1_no_018.csv",encoding="big5")
    
np_data = all_df.to_numpy()
    
    
sample = np.empty([18, 50])
for times in range(25):
    sample[:,times*2:(2*times)+2]=np_data[times*18:(18*times)+18,:]
    
with open('C:/Users/a1033/OneDrive/桌面/專題/new-test_mlp/MLP_H1N1_no_018_trans.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(sample)



