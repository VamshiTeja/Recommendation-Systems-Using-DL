# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-02-04 00:39:54
# @Last Modified by:   vamshi
# @Last Modified time: 2018-02-23 01:42:12

import sys
sys.path.append("../")
import dateutil.parser
import pickle
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import mmap
from tqdm import tqdm


def hour2vec(secs):
	'''
		To model the hours in a week
		There are 168 hours a week 
	'''
	hour = str(datetime.fromtimestamp(secs))[11:13] 
	day = datetime.fromtimestamp(secs).weekday()
	return int(day)*24+int(hour)



def load_batched_data(batch_size,b,e,d,g,target_gaps):
	'''
		Function to load data as batches of a user 
	'''
	#print(b.shape[0])
	randIxs = np.random.permutation(len(b))
	start,end =0,batch_size
	
	maxLength = 0
	for inp in b:
		maxLength = max(maxLength, inp.shape[0])
	#print(maxLength)

	while(end<=b.shape[0]):
		batchSeqLengths = np.zeros(batch_size)
		
		for batchI, origI in enumerate(randIxs[start:end]):
			batchSeqLengths[batchI] = b[origI].shape[-1]
	
		batchInputs_b = np.zeros((batch_size, maxLength))
		batchInputs_e = np.zeros((batch_size, maxLength))
		batchInputs_d = np.zeros((batch_size, maxLength))
		batchInputs_g = np.zeros((batch_size, maxLength))
		batchTargetList = []
		
		for batchI, origI in enumerate(randIxs[start:end]):
			padSecs = maxLength - b[origI].shape[0]
			batchInputs_b[batchI,:] = np.pad(b[origI], ((0),(padSecs)), 'constant', constant_values=0)
			batchInputs_e[batchI,:] = np.pad(e[origI], ((0),(padSecs)), 'constant', constant_values=0)
			batchInputs_d[batchI,:] = np.pad(d[origI], ((0),(padSecs)), 'constant', constant_values=0)
			batchInputs_g[batchI,:] = np.pad(g[origI], ((0),(padSecs)), 'constant', constant_values=0)
			batchTargetList.append(target_gaps[origI])

		start += batch_size
		end += batch_size
		batchInputs_d = np.vstack(batchInputs_d)
		batchInputs_b = np.vstack(batchInputs_b)
		batchInputs_e = np.vstack(batchInputs_e)
		batchInputs_g = np.vstack(batchInputs_g)
		#print(batchInputs_g.shape)
		#sbatchTargetList = np.vstack(batchTargetList)
		#print(batchInputs_g.shape)
		yield (batchInputs_b,batchInputs_e,batchInputs_d,batchInputs_g,batchTargetList,batchSeqLengths)

def load_batched_data_test(maxLength,batch_size,b,e,d,g,target_gaps):
	'''
		Function to load data as batches of a user 
	'''
	#print(b.shape[0])
	randIxs = np.random.permutation(len(b))
	start,end =0,batch_size

	while(end<=b.shape[0]):
		batchSeqLengths = np.zeros(batch_size)
		
		for batchI, origI in enumerate(randIxs[start:end]):
			batchSeqLengths[batchI] = b[origI].shape[-1]
	
		batchInputs_b = np.zeros((batch_size, maxLength))
		batchInputs_e = np.zeros((batch_size, maxLength))
		batchInputs_d = np.zeros((batch_size, maxLength))
		batchInputs_g = np.zeros((batch_size, maxLength))
		batchTargetList = []
		
		for batchI, origI in enumerate(randIxs[start:end]):
			padSecs = maxLength - b[origI].shape[0]
			batchInputs_b[batchI,:] = np.pad(b[origI], ((0),(padSecs)), 'constant', constant_values=0)
			batchInputs_e[batchI,:] = np.pad(e[origI], ((0),(padSecs)), 'constant', constant_values=0)
			batchInputs_d[batchI,:] = np.pad(d[origI], ((0),(padSecs)), 'constant', constant_values=0)
			batchInputs_g[batchI,:] = np.pad(g[origI], ((0),(padSecs)), 'constant', constant_values=0)
			batchTargetList.append(target_gaps[origI])

		start += batch_size
		end += batch_size
		batchInputs_d = np.vstack(batchInputs_d)
		batchInputs_b = np.vstack(batchInputs_b)
		batchInputs_e = np.vstack(batchInputs_e)
		batchInputs_g = np.vstack(batchInputs_g)
		#print(batchInputs_g.shape)
		#sbatchTargetList = np.vstack(batchTargetList)
		#print(batchInputs_g.shape)
		yield (batchInputs_b,batchInputs_e,batchInputs_d,batchInputs_g,batchTargetList,batchSeqLengths)