# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-02-23 01:21:30
# @Last Modified by:   vamshi
# @Last Modified time: 2018-03-01 20:59:09
import os 
import sys


for i in range(343):
	print"Training user ", i+1
	os.system("python train.py " + str(i))