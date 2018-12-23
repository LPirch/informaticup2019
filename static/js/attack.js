$$$ = function(a) { return document.getElementById(a); }

function attackChanged(option) {
	options = $$$("attack-options").children;

	for (var i = 0; i < options.length; i++) {
		if (options[i].id.startsWith(option.value)) {
			options[i].style.display = "block";
		} else {
			options[i].style.display = "none";
		}
	}
}