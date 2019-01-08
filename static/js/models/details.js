document.addEventListener("DOMContentLoaded", function() {
	pollConsole();
});

function pollConsole() {
	var interval = setInterval(fetch, 1000);
	fetch();

	function fetch() {
		var xhttp = new XMLHttpRequest();
		xhttp.onreadystatechange = function() {
			if (this.readyState == 4 && this.status == 200) {
				var data = JSON.parse(xhttp.responseText);

				var consoleDiv = document.getElementById("console");
				consoleDiv.innerHTML = data["console"];
				consoleDiv.scrollTop = consoleDiv.scrollHeight;

				if (!data["running"]) {
					clearInterval(interval);
					showFinalize();
				} else {
					document.querySelector('#abort').classList.add('show');
				}
			}
		}
		xhttp.open("GET", "models/proc_info", true);
		xhttp.send();
	}
}

function showFinalize() {
	var abort = document.querySelector('#abort');
	var clear = document.querySelector('#clear');

	abort.classList.remove('show');
	clear.classList.add('show');
}