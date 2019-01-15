import datetime

'''
Simple utility class for measuring
the run time of a block:

with Timer("mytimer"):
	foo()
'''
class Timer:
	def __init__(self, name):
		self.name = name

	def __enter__(self):
		self.time = datetime.datetime.now()

	def __exit__(self, type, value, traceback):
		print(self.name, datetime.datetime.now() - self.time)
