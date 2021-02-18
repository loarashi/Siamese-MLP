# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 12:41:56 2020

@author: a1033
"""
'''
h1n1
1 4 5 14 15 16 19 22 23 24(176) 2 6 7 8 9 10 12 13 17 20(192)  3 21(180) 11(154) 18(198) 

1 2 4 5 6 7 8 9 10 12 13 14 15 16 17 19 20 22 23 24(16)   3 21(15)    11(14)   18(18)

h3n2
1 5 6 7 10 12 15(128) 2 3 9 11 13 14 16(112) 4 17(105) 8(120)

1 2 3 5 6 7 9 10 11 12 14 15 16 (16)   4 8 17(15)   13(14)

sars
1 3(275) 2(270) 4(155) 5 6(324) 7 8(288) 9(72) 10 11(36)
'''
import csv

tp=[]
tn=[]
fp=[]
fn=[]
all_list=[]
a=0
b=0
c=0
d=0


with open('C:/Users/a1033/OneDrive/桌面/專題/new-test_mlp/MLP_H3N2_no_013.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([0,0])
        
for j in range(1,26):

    
    fp = open("C:/Users/a1033/OneDrive/桌面/專題/new-test_mlp/MLP_H3N2_no_013_result_"+str(j)+".txt",'r')
 

    all_lines = fp.readlines()
    all_list=all_lines
    
    

    with open('C:/Users/a1033/OneDrive/桌面/專題/new-test_mlp/MLP_H3N2_no_013.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
                        
        for k in range(14):
            a=int(all_list[k+4][7])
            b=int(all_list[k+18][11])*1+int(all_list[k+18][13])*0.1+int(all_list[k+18][14])*0.01+int(all_list[k+18][15])*0.001+int(all_list[k+18][16])*0.0001+int(all_list[k+18][17])*0.00001+int(all_list[k+18][18])*0.000001
            writer.writerow([a,b])                
                


            
            
            
            
            
            
            
            
            
            
            
            
            
            