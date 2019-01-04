$$$ = function(a) { return document.getElementById(a); }

function trainingChanged(option) {
	options = $$$("training-options").children;

	for (var i = 0; i < options.length; i++) {
		if (options[i].id.startsWith(option.value)) {
			options[i].style.display = "block";
		} else {
			options[i].style.display = "none";
		}
	}
}

function proc_delete(pid) {
	var xhttp = new XMLHttpRequest();
	xhttp.open("GET", "models/proc_delete?pid="+pid, true);
	xhttp.send();
}
