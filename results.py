# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2018-02-23 15:03:03
# @Last Modified by:   vamshi
# @Last Modified time: 2018-03-02 12:23:34

import os
import sys
import numpy as np
import math

res_dir = "./main/results/"
rmse_all = []
mae_all = []

for i in range(992):
	mae_file = res_dir + str(i) + "_mae.npy"
	rmse_file = res_dir + str(i) + "_rmse.npy"
	if(os.path.exists(mae_file) & os.path.exists(mae_file)):
		mae_usr = np.load(mae_file)
		rmse_user = np.load(rmse_file)
		mae = np.amin(mae_usr)
		rmse = np.amin(rmse_user)

		if(math.isnan(mae) | math.isnan(rmse)):
			continue
		else:
			rmse_all.append(rmse)
			mae_all.append(mae)

print (len(mae_all),len(rmse_all))
print np.mean(rmse_all),np.mean(mae_all)
	
