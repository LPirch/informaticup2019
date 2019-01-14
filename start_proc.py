import os
import sys
import subprocess

if __name__ == '__main__':
	""" Starts a subprocess, returns its PID and immediately exists (used for double fork).
		Usage: python start_proc.py [proc_dir] [cmd] [arg1] .. [argN]
	"""
	args = sys.argv
	assert len(args) >= 2

	# ignore first arg (name of script itself)
	args.pop(0)

	proc_dir = args.pop(0)

	os.setsid()
	with open(os.path.join(proc_dir, 'stdout'), 'wb') as f:
		p = subprocess.Popen(args, stdout=f, stderr=f, bufsize=1, universal_newlines=True)
	
	print(p.pid)