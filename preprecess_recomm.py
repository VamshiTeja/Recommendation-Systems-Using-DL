# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:02:31 2017

@author: sherin
"""

import dateutil.parser
import pickle
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import mmap
from tqdm import tqdm
runtime = time.time()
dataset = "lastfm"
DATASET_DIR = "/media/sherin/New Volume1/Thesis/spyder_lgcp/lgcp/"
#DATASET_FILE = '/home/sherin/Downloads/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
DATASET_FILE = './data/sample.tsv'
DATASET_W_CONVERTED_TIMESTAMPS = DATASET_DIR + '/1_converted_timestamps.tsv'
dataset_list = []

dataset_list = []


def get_num_lines(file_path):
    print file_path
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines
    
artist_map = {}
user_map = {}
artist_id = ''
user_id = ''    

print "Reading from file..."
data = pd.read_csv(DATASET_FILE, header = None,sep='\t',names=['user_event','timestamp','art_id','art_name','track_id','track_name'])

print"Mapping users to unique ids..."
user_event = data.user_event.unique()
user_dict = dict(zip(user_event, range(len(user_event))))
df = data.applymap(lambda s: user_dict.get(s) if s in user_dict else s)

print"Mapping artists to unique ids..."
art_id= df.art_id.unique()
art_dict = dict(zip(art_id, range(len(art_id))))


df = df.applymap(lambda s: art_dict.get(s) if s in art_dict else s)
df = df.drop('art_name', 1)
df = df.drop('track_id', 1)
df = df.drop('track_name', 1)

columnsTitles=["timestamp","user_event","art_id"]
df=df.reindex(columns=columnsTitles)

df.insert(loc=0, column='STR', value=1)
df.insert(loc=4, column='is_rumour', value=1)

print"Processing timestamp columns..."
df['timestamp'] = df.apply(lambda row:time.mktime(datetime.strptime(row[1], "%Y-%m-%dT%H:%M:%SZ").timetuple()) , axis=1)
'''with open(DATASET_FILE, 'rt', buffering=10000) as dataset:
    for line in tqdm(dataset, total=get_num_lines(DATASET_FILE)):

        line = line.split('\t')
        user_id     = line[0]
        
        if user_id not in user_map:
            user_map[user_id] = len(user_map)
        
        timestamp   = time.mktime(datetime.strptime(line[1], "%Y-%m-%dT%H:%M:%SZ").timetuple())
        #timestamp = line[1]
        artist_id   = line[2]
        
        if artist_id not in artist_map:
            artist_map[artist_id] = len(artist_map)
        
        dataset_list.insert(0, ['0', timestamp, user_map[user_id],artist_map[artist_id],'1'] )
    '''    

'''for i in range(len(dataset_list)):
    user_id = dataset_list[i][2]
    artist_id = dataset_list[i][3]
    if user_id not in user_map:
        user_map[user_id] = len(user_map)
    if artist_id not in artist_map:
        artist_map[artist_id] = len(artist_map)
    dataset_list[i][2] = user_map[user_id]
    dataset_list[i][3] = artist_map[artist_id]

'''

#df = pd.DataFrame(dataset_list, columns=['STR','timestamp','user_event','art_id','is_rumour'])
print '***RAW DATA STATISTICS***'
print 'Number of users :', len(np.unique(df['user_event']))
print 'Number of artists :',len(np.unique(df['art_id']))
print 'Number of lines :',len(df)
print '\n'




print 'Removing unpopular artists/retaininig top 20000 artists...'
#Remove unpopular items
artist_popularity = df['art_id'].groupby(df['art_id']).size().nlargest(20000)
df = df[np.in1d(df.art_id, artist_popularity.index)]


#Remove inactive users a first time
#print 'Removing inactive users' 
#user_activity = df.groupby('user_event').size()
#df = df[np.in1d(df.user_event, user_activity[user_activity >= 1000].index)]

print '\n'
print '***PROCESSED DATA STATISTICS***'
print 'Number of users :', len(np.unique(df['user_event']))
print 'Number of artists :',len(np.unique(df['art_id']))
print 'Number of lines :',len(df)

print '\n'

print('Sort data in chronological order...')
df = df.sort_values(['user_event','timestamp'], ascending=[True,True])

print '\n'
print 'Splitting into sessions...'


time_diff =  df['timestamp'] - df['timestamp'].shift(1)
time_diff[len(time_diff)-1] = 60*60

df = df[np.in1d(df.index, time_diff[time_diff>=60*60].index)]

#df = df[np.in1d(df.index,df[df.timestamp<=(6*30*24*60*60*2)].index)]
print 'Number of session : ',len(df)
print '\n'

#Remove inactive users a first time
print 'Removing inactive users...' 
user_activity = df.groupby('user_event').size()
df = df[np.in1d(df.user_event, user_activity[user_activity >= 500].index)]
print 'Number of users :', len(np.unique(df['user_event']))
print 'Number of artists :',len(np.unique(df['art_id']))
print 'Number of lines :',len(df)
'''
a = df.index.tolist()
b = list(map(lambda x: l-1-x, a))
c = [t - s for s, t in zip(b, b[1:])]
c.append(l-c[-1])
se = pd.Series(c)
df['session_count'] = se.values
df = df[np.in1d(df.index,df[df.session_count>=50].index)]
df = df.drop('session_count', 1)
'''
print 'Number of session : ',len(df)
df.to_csv('list.csv', index=False,sep=' ')

