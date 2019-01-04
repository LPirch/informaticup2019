from project_conf import MODEL_SAVE_PATH

import subprocess
import os.path

class CWL2AttackHandler:
	def parse_arguments(request):
		return {
			"binary_search_steps": str(int(request.POST["binary_search_steps"])),
			"confidence": str(int(request.POST["confidence"])),
			"max_iterations": str(int(request.POST["max_iterations"])),
			"target": str(int(request.POST["target"])),
			"image": request.FILES["imagefile"],
			"attack": "cwl2",
			"model": str(request.POST["modelname"]),
			"model_folder": MODEL_SAVE_PATH
		}
	
	def start(process_dir, kwargs):
		popen_args = []
		for k,v in kwargs.items():
			popen_args.append('--'+k)
			if v:
				popen_args.append(str(v))
		
		popen_args = ["python", "start_proc.py", process_dir, "python", "attack_model.py"] + popen_args
		print("="*80)
		print(popen_args)
		pid = int(subprocess.check_output(popen_args).strip())
		
		return pid