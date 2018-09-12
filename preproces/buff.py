def split_sessions(sessions,gaps):
	'''
		make sessions from a user history
	
	'''
	
	num_samples = 100
	min_samples_test = 50 
	new_sessions = []
	new_begin,new_end,new_d,new_gaps = [],[],[],[]
	target_gaps = []
	num_sessions_user = len(gaps)
	min_num_sessions = 50
	min_time = sessions[0][0]
	print(min_time)
	num = 0
	while(num<num_samples):
		sess =[]
		b,e  =[],[]
		d = []
		g = []
		start_sess = np.random.randint(0,num_sessions_user-2*min_num_sessions)
		end_sess   = np.random.randint(start_sess+min_num_sessions, num_sessions_user)

		start_sess_test = np.random.randint(0,num_sessions_user-2*min_num_sessions)
		end_sess_test   = np.random.randint(start_sess+min_num_sessions, num_sessions_user)
		
		if(int(gaps[end_sess-1])/1800 !=0):
			for j in range(start_sess+1,end_sess):
				sess.append(sessions[j])
				b.append(int(sessions[j][0]-min_time)/3600)
				e.append(int(sessions[j][1]-min_time)/3600)
				g.append((sessions[j][0]-sessions[j-1][1])/1800)
				d.append(hour2vec(int(sessions[j][1])))
			g[0] = 0
			#vstack all lists
			new_sessions.append(np.array(sess))
			new_begin.append(np.array(b))
			new_end.append(np.array(e))
			new_d.append(np.array(d))
			new_gaps.append(np.array(g))
			target_gaps.append(int(gaps[end_sess-1])/1800)

			num += 1
	print(len(new_sessions),len(target_gaps))
	return np.array(new_sessions),np.array(new_begin),np.array(new_end),np.array(new_d),np.array(new_gaps),np.array(target_gaps)
