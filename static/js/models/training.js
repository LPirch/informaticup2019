$$$ = function(a) { return document.getElementById(a); }

document.addEventListener("DOMContentLoaded", function() {
	trainingChanged('');
});

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