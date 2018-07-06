#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 19:16:57 2018

@author: cadu
"""
import os
import json
import gzip
import numpy as np
filelist = os.listdir('2018-05-28')
for item in filelist:
    print(item)
def calc_ent(x):
    ent = 0
    x_arr = np.array(x)
    _sum = np.sum(x_arr)
    if _sum==0:
        return 0
    for ele in x:
        if ele==0:
            continue
        p=ele/_sum
        logp = np.log2(p)
        ent -= p * logp
    return ent
entropy = []
std = []
with gzip.open("2018-05-28/out2018-04-04_win16.gz","r") as fp:
    try:
        for line in fp:
            try:
                tmp = json.loads(line)
            except Exception as e:
                print(e)
                continue
            if 'version' not in tmp:
                #print(tmp['te']-tmp['ts'])
                print('mean packet length:',np.sum(tmp['bd'])/(len(tmp['packets'])+1)) #mean packet length
                entropy.append(calc_ent(tmp['bd'])) # entropy 
                std.append(np.std(tmp['bd'])) # stdandard devitation
                
                sec_bytes_l=[]
                sec_bytes=np.sum(tmp['bd'])/(tmp['te']-tmp['ts'])
                if sec_bytes=='NaN' or sec_bytes=='inf':
                    sec_bytes=np.mean(sec_bytes_l)
                sec_bytes_l.append(sec_bytes)
                print('sec_bytes',sec_bytes)
                byte_l=[] # byte length in the packet
                if len(tmp['packets'])!=0:
                    print('len of packet',len(tmp['packets']))
                    for item in tmp['packets']:
                        print('item_b:',item['b'])
                        
                        byte_l.append(item['b'])
                        #min_pkl=tmp['']
                if len(byte_l)!=0:
                    print('max length',np.max(byte_l))
                else:
                    print('max length',np.mean(byte_l))
                print('#'*20)
    except Exception as e:
        print(e)
   
print("done")
calc_ent([1,1])
