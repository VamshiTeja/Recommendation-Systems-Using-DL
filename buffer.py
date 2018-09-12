
	num = 0
	while(num<num_samples):
		sess =[]
		b,e  =[],[]
		d = []
		g = []
		#test
		sess_t =[]
		b_t,e_t  =[],[]
		d_t = []
		g_t = []

		start_sess = np.random.randint(0,num_sessions_user-2*min_num_sessions)
		end_sess   = np.random.randint(start_sess+min_num_sessions, num_sessions_user)

		
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