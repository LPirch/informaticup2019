from project_conf import PROCESS_DIR, IMG_TMP_DIR, RANDOM_SEED

import os
import os.path
import subprocess
import random

random.seed(RANDOM_SEED)

def init_directories():
	if not os.path.exists(PROCESS_DIR):
		os.makedirs(PROCESS_DIR)

def get_token_from_pid(pid):
	pid = str(int(pid))

	if not os.path.exists(os.path.join(PROCESS_DIR, pid)):
		raise ValueError("unknown PID: "+ pid)

	try:
		with open(os.path.join(PROCESS_DIR, pid), "r") as f:
			token = f.read().strip()
	except:
		raise RuntimeError("Could not read from process pid-file: "+os.path.join(PROCESS_DIR, pid))

	return token

def is_pid_running(pid):
	""" Check For the existence of a unix pid. """
	try:
		os.kill(int(pid), 0)
	except OSError:
		return False
	else:
		return True

def get_running_procs(prefix=None):
	processes  = []
	for p in filter(lambda x: x.isdigit(), os.listdir(PROCESS_DIR)):
		# optionally filter associated token for given prefix
		if prefix and prefix not in get_token_from_pid(p):
			continue

		processes.append({
			"id": p,
			"running": is_pid_running(int(p))
	})
	return processes

def gen_token(prefix):
	return prefix+str(random.random())

def write_pid(token, pid):
	with open(os.path.join(PROCESS_DIR, str(pid)), 'w') as f:
		f.write(token)
	
def kill_proc(pid):
	pid = str(int(pid))

	if is_pid_running(pid):
		os.kill(int(pid), 9)
	
	try:
		token = get_token_from_pid(pid)
		if os.path.exists(os.path.join(PROCESS_DIR, token)):
			# clean stdout file
			if os.path.exists(os.path.join(PROCESS_DIR, token, "stdout")):
				os.remove(os.path.join(PROCESS_DIR, token, "stdout"))
			# clean token dir
			os.removedirs(os.path.join(PROCESS_DIR, token))
		
		if os.path.exists(os.path.join(PROCESS_DIR, pid)):
			# clean pid file
			os.remove(os.path.join(PROCESS_DIR, pid))
	except ValueError:
		pass
