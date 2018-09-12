# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-02-03 23:51:41
# @Last Modified by:   vamshi
# @Last Modified time: 2018-02-04 23:46:37

import dateutil.parser
import pickle
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import mmap
from tqdm import tqdm


months = dict(zip(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],range(12)))

USER_FILE = '../data/lastfm/lastfm-dataset-1K/userid-profile.tsv'
SAVE_DIR = "../data/lastfm/lastfm-dataset-1K/"

print "Reading users file"
user_data = pd.read_csv(USER_FILE,header =0, sep='\t',names=['id','gender','age','country','registered'])

print"Mapping users to unique ids..."
user_event = user_data.id.unique()
user_dict = dict(zip(user_event, range(len(user_event))))
df = user_data.applymap(lambda s: user_dict.get(s) if s in user_dict else s)
user_data = user_data.applymap(lambda s: user_dict.get(s) if s in user_dict else s)

print "Mapping countries to unique ids"
coun = user_data.country.unique()
coun_dict = dict(zip(coun,range(len(coun))))
df = user_data.applymap(lambda s: coun_dict.get(s) if s in coun_dict else s)

mean_age = user_data['age'].mean()
print mean_age

print"Processing registered columns..."

def convert_date(date):
	'''
	    converts to secs
		expects in format Aug 15, 2006
	'''

	date = str(date)

	if(date=='3'):
		return time.mktime((2006,8,15,0,0,0,0,0,0))
	a = date.split()
	mon = months[a[0]]
	day = int(a[1].split(',')[0])
	year = int(a[2])
	tim = (year,mon,day,0,0,0,0,0,0)
	secs = time.mktime(tim)
	return secs


def random():
	return np.random.randint(0,2)
def is_nan(x):
	return x!=x

def map_gender(g):
	'''
		map males to 1 and females to 0 else randomly assign
	'''
	if(g=='m'):
		return 1
	elif(g=='f'):
		return 0
	else:
		return random()
#print convert_date("Aug 15, 2006")

for index, row in df.iterrows():
	if(row[2]==None):
		row[2] = mean_age
	if(len(str(row[4])) == 14 or len(str(row[4]))== 13):
		row[4] = convert_date(row[4])


#a = lambda row: convert_date(row[4])
#print a(df['registered'])
df['registered'] = df.apply(lambda row: "Aug 15, 2006" if ((row[4]))==3 is None else row[4]  , axis=1)
df['registered'] = df.apply(lambda row: convert_date(row[4])  , axis=1)
#df['registered'] = df.apply(lambda row:time.mktime(datetime.strptime(row[1], "%Y-%m-%d").timetuple()) , axis=1)
#print user_data_df]


#set age to mean age for those not specified
df['age'] = df.apply(lambda row: mean_age if(is_nan(row[2])) else row[2], axis=1)
#map gender
df['gender'] = df.apply(lambda row: map_gender(row[3]), axis=1)

df = df.drop(['id'],axis=1)
df = df.astype(int)

user_data_mat = df.as_matrix()
print user_data_mat

#save user vectors 
save_file = SAVE_DIR + "user_data"
np.savez(save_file,)




