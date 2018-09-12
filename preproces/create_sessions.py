# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-01-27 17:02:56
# @Last Modified by:   vamshi
# @Last Modified time: 2018-06-12 23:09:01
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
import pickle
#from utils import hour2vec

DATASET_FILE = "../data/lastfm/lastfm-dataset-1K/processed.csv"

SAVE_DIR = "../data/lastfm/lastfm-dataset-1K/"

print "Reading from file..."
df = pd.read_csv(DATASET_FILE,sep=',',header=0)
#print df
#df = df.drop(['STR','is_rumour'],1)
#df = df.drop('is_rumour',1)

data_matrix = df.as_matrix()
print data_matrix.shape
l = data_matrix.shape[0]
timestamp = np.reshape(data_matrix[:,1],(l,1))
user_id = np.reshape(data_matrix[:,2],(l,1))
art_id = np.reshape(data_matrix[:,3],(l,1))

dataset = np.concatenate((user_id, timestamp, art_id), axis=1)
df = pd.DataFrame(dataset,columns=['user_id','timestamp','art_id'])
df = df.astype(int)

#create sessions and gaps user wise
#sessions : Nested list containing user_sessions where each user_session tuples of sessions
#models sessions less than one hours

def hour2vec(secs):
	'''
		To model the hours in a week
		There are 168 hours a week 
	'''
	hour = str(datetime.fromtimestamp(secs))[11:13] 
	day = datetime.fromtimestamp(secs).weekday()
	return int(day)*24+int(hour)


#function to get gaps
def get_sess_gaps():
	sessions = []
	gaps = []
	num_users = len(np.unique(user_id))
	for user in range(num_users):
		user_sessions = []
		user_gaps =[]
		user_rows = df.loc[df['user_id']==user].as_matrix()
		#print user_rows
		num_rows = user_rows.shape[0]
		act = []
		st_time = 0
		end_time = 0
		while(end_time<num_rows-1):
			if(float(user_rows[end_time+1,1]-user_rows[end_time,1])/3600.0>1):
				sess = [float(user_rows[st_time,1]),float(user_rows[end_time,1])]
				user_sessions.append(sess)
				act=[]
				st_time = end_time+1
				if(st_time<num_rows-1):        
					gap = user_rows[st_time,1] - user_rows[end_time,1]
					user_gaps.append(gap)
				end_time = st_time + 1
			else:
				#act.append(user_rows[end_time,2])
				end_time = end_time+1

		sessions.append(user_sessions)
		gaps.append(user_gaps)
	return sessions, gaps

#print (sess_user)
sessions,gaps = get_sess_gaps()
#print len(sessions[1]),len(gaps[1])
le = 0
for i in range(len(sessions)):
	le = le + len(sessions[i])

print le

#function to split into sessions
def split_sessions(sessions,gaps):
	'''
		make sessions from a user history
	'''	
	new_begin,new_end,new_d,new_gaps     = [],[],[],[]
	test_begin,test_end,test_d,test_gaps = [],[],[],[]
	train_target_gaps = []
	test_target_gaps  = []

	num_sessions_user = len(gaps)
	
	min_num_sessions = 100
	min_num_test_sessions = 50

	observation_window = 90.5*24  #90 days
	observation_window_test = 31*24 #30 days

	train_start = 0
	sess_number = 0
	print("max",num_sessions_user)
	while(sess_number<num_sessions_user):
		#train
		b,e,g,d =[],[],[],[]
		#test
		b_t,e_t,d_t,g_t =[],[],[],[]

		min_time = sessions[train_start][0]
		#print("train",int(sessions[sess_number][1]-sessions[train_start][0])/3600)
		while((sess_number<num_sessions_user) and int(sessions[sess_number][0]-sessions[train_start][0])/3600 <=observation_window):
			b.append(int(sessions[sess_number][0]-min_time)/3600)
			g.append(gaps[sess_number]/3600)
			e.append(int(sessions[sess_number][1]-min_time)/3600)
			d.append(hour2vec(int(sessions[sess_number][1])))
			sess_number += 1

		train_sess = sess_number
		
		if(sess_number+1<num_sessions_user):
			test_start = sess_number
			min_time = sessions[test_start][0]
			sess_number += 1
			#print((sessions[sess_number][1]-sessions[test_start][0])/3600)
			while(sess_number+1<num_sessions_user and int(sessions[sess_number][0]-sessions[test_start][0])/3600 <= observation_window_test):
				b_t.append(int(sessions[sess_number][0]-min_time)/3600)
				g_t.append(int(gaps[sess_number-1])/3600)
				e_t.append(int(sessions[sess_number][1]-min_time)/3600)
				d_t.append(hour2vec(int(sessions[sess_number][1])))
				sess_number += 1

			if(len(b)>=100 and len(b_t)>=40):
				new_begin.append(np.array(b))
				new_end.append(np.array(e))
				new_d.append(np.array(d))
				new_gaps.append(np.array(g))		
				train_target_gaps.append(int(gaps[train_sess])/3600)

				test_begin.append(np.array(b_t))
				test_end.append(np.array(e_t))
				test_d.append(np.array(d_t))
				test_gaps.append(np.array(g_t))
				test_target_gaps.append(int(gaps[sess_number])/3600)
			train_start = sess_number
		else:
			return ((np.array(new_begin),np.array(new_end),np.array(new_d),np.array(new_gaps),np.array(train_target_gaps)),
					(np.array(test_begin),np.array(test_end),np.array(test_d),np.array(test_gaps),np.array(test_target_gaps)))	
		
	return ((np.array(new_begin),np.array(new_end),np.array(new_d),np.array(new_gaps),np.array(train_target_gaps)),
	(np.array(test_begin),np.array(test_end),np.array(test_d),np.array(test_gaps),np.array(test_target_gaps)))

sessions, gaps = get_sess_gaps() 

def save_all_users():	
	num_sessions = 0
	test_sessions = 0
	user,cnt = 0, 0
	while(user<len(sessions)):
		print"For user ", user+1
		sess_user,gaps_user = sessions[user],gaps[user]

		res = split_sessions(sess_user, gaps_user)
		train, test = res
		b,e,d,new_gaps,target_gaps = train
		b_t,e_t,d_t,new_gaps_t,target_gaps_t = test
		if(len(b)==0 and len(b_t)==0):
			user += 1
			continue
		for i in range(len(new_gaps)):
			num_sessions += len(new_gaps[i])
		for i in range(len(new_gaps_t)):
			test_sessions += len(new_gaps_t[i])
		#print(type(new_gaps[0]))
		save_as_pickle = "../data/dataset_" +str(cnt)+"_train.pickle"

		try:
			f = open(save_as_pickle,'wb')
			save = {
			'b'   : b,
			'e'   : e,
			'd'   : d,
			'g'   : new_gaps,
			'target_gaps'   : target_gaps,
			}
			pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
			f.close()
		except Exception as e:
			print('Unable to save data to ',save_as_pickle,':',e)
			raise

		statinfo = os.stat(save_as_pickle)
		print('Compressed pickle size',statinfo.st_size)
		save_as_pickle = "../data/dataset_" +str(cnt)+"_test.pickle"

		try:
			f = open(save_as_pickle,'wb')
			save = {
			'b'   : b_t,
			'e'   : e_t,
			'd'   : d_t,
			'g'   : new_gaps_t,
			'target_gaps'   : target_gaps_t,
			}
			pickle.dump(save,f,pickle.HIGHEST_PROTOCOL)
			f.close()
		except Exception as e:
			print('Unable to save data to ',save_as_pickle,':',e)
			raise

		statinfo = os.stat(save_as_pickle)
		print('Compressed pickle size',statinfo.st_size)
		user += 1
		cnt += 1
	print(cnt)
	print(num_sessions)
	print(test_sessions)


save_all_users()


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
		batchTargetList = np.zeros((batch_size))
		
		for batchI, origI in enumerate(randIxs[start:end]):
			padSecs = maxLength - b[origI].shape[0]
			batchInputs_b[batchI,:] = np.pad(b[origI], ((0),(padSecs)), 'constant', constant_values=0)
			batchInputs_e[batchI,:] = np.pad(e[origI], ((0),(padSecs)), 'constant', constant_values=0)
			batchInputs_d[batchI,:] = np.pad(d[origI], ((0),(padSecs)), 'constant', constant_values=0)
			batchInputs_g[batchI,:] = np.pad(g[origI], ((0),(padSecs)), 'constant', constant_values=0)
			batchTargetList[batchI] = target_gaps[origI]

		start += batch_size
		end += batch_size
		batchInputs_d = np.vstack(batchInputs_d)
		batchInputs_b = np.vstack(batchInputs_b)
		batchInputs_e = np.vstack(batchInputs_e)
		batchInputs_g = np.vstack(batchInputs_g)
		#print(batchInputs_b[0].shape)
		#batchTargetList = np.vstack(batchTargetList)
		#print(batchInputs_g[0].shape)
		yield (batchInputs_b,batchInputs_e,batchInputs_d,batchInputs_g,batchTargetList,batchSeqLengths)


