import subprocess
import os.path

class TrainHandlerRebuild:
	def parse_arguments(request):
		kwargs =  {
			'modelname': str(request.POST["modelname"]),
			'epochs': str(request.POST["epochs"]),
			'batch_size': str(request.POST["batch_size"]),
			'learning_rate': str(request.POST["lr"]),
			'optimizer': str(request.POST["optimizer"]),
			'dataset': str(request.POST["dataset"]),
			'validation_split': str(request.POST["valsplit"]),
			'max_per_class': str(request.POST["maxperclass"]),
			'enable_tensorboard': str(request.POST["enable_tensorboard"]),
			'keras_verbosity': 2
		}

		if int(str(request.POST["augmentation"])):
			kwargs.update({'load_augmented': None})
		if int(str(request.POST['tensorboard'])):
			kwargs.update({'enable_tensorboard': None})


		return kwargs

	def start(process_dir, kwargs):
		popen_args = []
		for k,v in kwargs.items():
			popen_args.append('--'+k)
			if v:
				popen_args.append(str(v))
		
		popen_args = ["python", "start_proc.py", process_dir, "python", "train_rebuild.py"] + popen_args
		pid = int(subprocess.check_output(popen_args).strip())
		
		return pid

class TrainHandlerSubstitute:
	def parse_arguments(request):
		kwargs =  {
			"modelname": str(request.POST["modelname"]),
			"enable_tensorboard": str(request.POST["enable_tensorboard"]),
			"lmbda": float(str(request.POST["lmbda"])),
			"tau": int(str(request.POST["tau"])),
			"n_jac_iteration": int(str(request.POST["n_jac_iteration"])),
			"n_per_class": int(str(request.POST["n_per_class"])),
			"batch_size": int(str(request.POST["batch_size"])),
			"descent_only":  bool(str(request.POST["descent_only"]))
		}

		return kwargs

	def start(process_dir, kwargs):
		popen_args = []
		for k,v in kwargs.items():
			popen_args.append('--'+k)
			if v:
				popen_args.append(str(v))
		
		popen_args = ["python", "start_proc.py", process_dir, "python", "train_substitute.py"] + popen_args
		pid = int(subprocess.check_output(popen_args).strip())
		
		return pid