# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-02-04 12:23:05
# @Last Modified by:   vamshi
# @Last Modified time: 2018-03-01 09:53:41
import sys
sys.path.append("../model")
sys.path.append("../utils")
import dateutil.parser
import pickle
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import mmap
from tqdm import tqdm
from utils import load_batched_data,load_batched_data_test
import tensorflow as tf
from model import Model
from sys import argv

dataset = "lastfm"
DATASET_FILE = '../data/lastfm/lastfm-dataset-1K/'
USER_FILE = '../data/lastfm/lastfm-dataset-1K/userid-profile.tsv'

train_dataset = "../data/dataset.pickle"
test_dataset = "../data/dataset_test.pickle"

mode = "train"

#Train settings
lr = 1e-4
dt = 1.0
T = 4320.0/2*dt
batch_size = 1
embed_dim = 64
lambdim = T
n_gaps = T
n_hidden = 64
isTraining = True
grad_clip = -1
num_epochs = 50

#Function to load data
def get_data(train_dataset):
	with open(train_dataset,'rb') as f:
			save = pickle.load(f)
			b = save['b']
			e = save['e']
			d = save['d']
			g = save['g']	
			target_gaps = save['target_gaps']
	return b, e, d, g,target_gaps

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class Runner(object):

	def _default_configs(self):
	  return {'batch_size': batch_size,
	  		  'T':T,
	  		  'dt':dt,
	  		  'embed_dim' : embed_dim,
	  		  'lambdim'   : lambdim,
	  		  'n_gaps'    : n_gaps,
			  'n_hidden': n_hidden,
			  'isTraining' : isTraining,
			  'learning_rate': lr,
			  'grad_clip': grad_clip,
			}

	def load_data(self, batch_size,b,e,d,target_gaps):
		return load_batched_data(batch_size,b,e,d,target_gaps)

	def getdist(self,lamb,sessLengths):
		dist = np.zeros(shape=(lamb.shape[0],lamb.shape[2]))
		num_batches = lamb.shape[0]
		summ = []
		for i in range(num_batches):
			seslen = int(sessLengths[i])
			lr = lamb[i][seslen-1]
			summ = []
			summ.append(lr[0])
			for j in range(lr.shape[0]):
				summ.append(summ[j-1]+lr[j])
			for j in range(lr.shape[0]):
				dist[i][j] = lr[j]*np.exp(-1*summ[j]*dt)

		return dist


	def predict_gaps(self,lamb,sessLengths,maxSessLen):
		predict_gap = []
		num_batches = lamb.shape[0]
		dist = self.getdist(lamb,sessLengths)

		for i in range(num_batches):
			s = int(sessLengths[i])
			lr = list(dist[i])
			#print(lr)
			#lr = normalize(lr)
			sums = []
			sums.append(lr[0])
			for j in range(1,len(lr)):
				sums.append(sums[j-1] + lr[j])

			for k in range(len(sums)):
				if(sums[k]>=0.5):
					break
			idx = k
			predict_gap.append(int(idx)*dt)

		return predict_gap


		 
	def MAE(self,predict_gap,target_gaps):
		predict_gap = np.array(predict_gap)
		target_gaps = np.array(target_gaps)
		return np.mean(np.abs(predict_gap-target_gaps))


	def MSE(self,predict_gap,target_gaps):
		predict_gap = np.array(predict_gap)
		target_gaps = np.array(target_gaps)
		return np.sqrt(np.mean((predict_gap - target_gaps)**2))


	def run(self,user):
		args_dict = self._default_configs()
		args = dotdict(args_dict)

		#get data
		#print("Loading data")
		train_dataset = "../data/dataset_"+str(user)+"_train.pickle"
		test_dataset = "../data/dataset_"+str(user)+"_test.pickle"
		b, e, d, g,target_gaps = get_data(train_dataset)
		b_t, e_t, d_t, g_t,target_gaps_t = get_data(test_dataset)
		#print(len(b))
		#print(type(g))
		#g = np.vstack(g)
		#print(b.shape)
		totalN = len(b)
		#print("b :",len(b))
		num_batches = len(b)/batch_size
		maxLength = 0
		for x in b:
			maxLength = max(maxLength, x.shape[0])
		maxSessLen = maxLength

		print("Building Model")
		recom = Model(args,maxSessLen)
		recom.build_graph(args, maxSessLen)
		print("Starting Session")
		#print(recom.config)
		test_Err = []
		test_MAE = []
		with tf.Session(graph=recom.graph) as sess:
			if(mode=='train'):
				writer = tf.summary.FileWriter("loggingdir",graph=recom.graph)
				sess.run(recom.initial_op)
				for epoch in range(num_epochs):
					# training
					start = time.time()
					print('Epoch {} ...'.format(epoch + 1))
					batchLoss = np.zeros(num_batches)
					batchErr = np.zeros(num_batches)
					batchErrMAE = np.zeros(num_batches)
					batchRandIxs = np.random.permutation(num_batches)
					for batch, batchOrigI in enumerate(batchRandIxs):
						batchInputs_b,batchInputs_e,batchInputs_d,batchInputs_g,batchTargetList,batchSeqLengths = next(load_batched_data(batch_size,b,e,d,g,target_gaps))
						#print(type(batchInputs_g))
						feedDict = {recom.inputb: batchInputs_b, recom.inpute: batchInputs_e,
									recom.inputg: batchInputs_g, recom.inputd: batchInputs_d,
									recom.target_gaps: batchTargetList,recom.sessLengths: batchSeqLengths}

						_, l,lamb = sess.run([recom.optimizer, recom.loss, recom.lamb],feed_dict=feedDict)
						#writer.add_summary(summary,epoch*num_batches+batch)

						batchLoss[batch] = l
						

						predicted_gaps = self.predict_gaps(lamb, batchSeqLengths, maxSessLen)
						error = self.MSE(predicted_gaps, batchTargetList)
						error_MAE = self.MAE(predicted_gaps, batchTargetList)
						batchErr[batch] = error
						batchErrMAE[batch] = error_MAE
						#print(len(batchTargetList))
						print('batch:{}/{},epoch:{}/{},train loss={:.3f},RMSE={:.3f},MAE={:.3f}'.format(
							batch+1, len(batchRandIxs), epoch+1, num_epochs, l,error,error_MAE))
				
						#print(predicted_gaps)
						#print(batchTargetList)
					
					print("MAE after %d epoch is %.2f"%(epoch+1,np.mean(batchErrMAE)))

					end = time.time()
					delta_time = end - start
					print('Average loss of the epoch is %.2f'%np.mean(batchLoss))
					print('Epoch ' + str(epoch + 1) + ' needs time:' + str(delta_time) + ' s')
					
					#Testing after ever epoch
					num_b = len(b_t)/batch_size
					batchErr = np.zeros(num_b)
					batchErrMAE = np.zeros(num_b)
					batchRandIxs = np.random.permutation(num_b)
					for batch, batchOrigI in enumerate(batchRandIxs):
						batchInputs_b,batchInputs_e,batchInputs_d,batchInputs_g,batchTargetList,batchSeqLengths = next(load_batched_data_test(maxSessLen,batch_size,b_t,e_t,d_t,g_t,target_gaps_t))
						feedDict = {recom.inputb: batchInputs_b, recom.inpute: batchInputs_e,
										recom.inputg: batchInputs_g, recom.inputd: batchInputs_d,
										recom.target_gaps: batchTargetList,recom.sessLengths: batchSeqLengths}
						__, l,lamb = sess.run([recom.optimizer, recom.loss, recom.lamb],feed_dict=feedDict)

						predicted_gaps = self.predict_gaps(lamb, batchSeqLengths, maxSessLen)
						error = self.MSE(predicted_gaps, batchTargetList)
						error_MAE = self.MAE(predicted_gaps, batchTargetList)
						batchErr[batch] = error
						batchErrMAE[batch] = error_MAE

					test_Err.append(np.mean(batchErr))
					test_MAE.append(np.mean(batchErrMAE))
					#print("RMSE error of test set is %.2f"% np.mean(batchErr))
					print("MAE error of test set is %.2f\n"% np.mean(batchErrMAE))
		return test_Err,test_MAE



if __name__ == '__main__':
	
	if(len(sys.argv)!=2):
		print"correct usage usr"
		sys.exit(1)
	else:
	    usr = sys.argv[1]
    	runner = Runner()
    	rmse,mae = runner.run(usr)
    	filename = "./results/"+str(usr)
    	np.save(filename+'_rmse', rmse)
    	np.save(filename+'_mae', mae)
