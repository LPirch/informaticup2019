import subprocess
import os.path

class CWL2AttackHandling:
    def handle_arguments(request):
        return {
            "bss": str(int(request.POST["cwl2_binary_search_steps"])),
            "confidence": str(int(request.POST["cwl2_confidence"])),
            "max_iterations": str(int(request.POST["cwl2_max_iterations"])),
            "target": str(int(request.POST["target"])),
            "image": request.FILES["imagefile"]
        }

    def start(outdir, process_dir, bss, confidence, max_iterations, target, src_img_path, **kwargs):
        with open(os.path.join(process_dir, "stdout"), "wb") as f:
            p = subprocess.Popen(["python3", "attack_model.py",
                "--attack", "cwl2",
                "--model", "gtsrb_model",
                "--model_folder", "model/trained/",
                "--outdir", outdir,
                "--binary_search_steps", bss,
                "--confidence", confidence,
                "--max_iterations", max_iterations,
                "--target", target,
                "--image", src_img_path], stdout=f, stderr=f, bufsize=1, universal_newlines=True)

            return p

class RobustCWL2AttackHandling:
    def handle_arguments(request):
        return {
            "bss": str(int(request.POST["robust_cwl2_binary_search_steps"])),
            "confidence": str(int(request.POST["robust_cwl2_confidence"])),
            "max_iterations": str(int(request.POST["robust_cwl2_max_iterations"])),
            "target": str(int(request.POST["target"])),
            "image": request.FILES["imagefile"]
        }

    def start(outdir, process_dir, bss, confidence, max_iterations, target, src_img_path, **kwargs):
        with open(os.path.join(process_dir, "stdout"), "wb") as f:
            p = subprocess.Popen(["python3", "attack_model.py",
                "--attack", "robust_cwl2",
                "--model", "gtsrb_model",
                "--model_folder", "model/trained/",
                "--outdir", outdir,
                "--binary_search_steps", bss,
                "--confidence", confidence,
                "--max_iterations", max_iterations,
                "--target", target,
                "--image", src_img_path], stdout=f, stderr=f, bufsize=1, universal_newlines=True)

            return p

class PhysicalAttackHandling:
    def handle_arguments(request):
        return {
            "max_iterations": str(int(request.POST["physical_max_iterations"])),
            "mask_image": request.FILES["physical_mask_image"],
            "target": str(int(request.POST["target"])),
            "image": request.FILES["imagefile"]
        }

    def start(outdir, process_dir, max_iterations, target, src_img_path, mask_path, **kwargs):
        with open(os.path.join(process_dir, "stdout"), "wb") as f:
            p = subprocess.Popen(["python3", "attack_model.py",
                "--attack", "physical",
                "--model", "gtsrb_model",
                "--model_folder", "model/trained/",
                "--outdir", outdir,
                "--max_iterations", max_iterations,
                "--target", target,
                "--mask_image", mask_path,
                "--image", src_img_path], stdout=f, stderr=f, bufsize=1, universal_newlines=True)

            return p