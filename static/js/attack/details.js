function pollConsole(pid) {
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
				}
			}
		}
		xhttp.open("GET", "attack/proc_info?pid=" + pid, true);
		xhttp.send();
	}
}

function pollImages(pid, newImgCallback) {
	// cache already loaded images
	var images = [];
	var interval = setInterval(fetch, 1000);
	fetch();

	function fetch() {
		var xhttp = new XMLHttpRequest();
		xhttp.onreadystatechange = function() {
			if (this.readyState == 4 && this.status == 200) {
				var data = JSON.parse(xhttp.responseText);

				var recv_images = data["images"];
				var new_images = []

				for (var i = 0; i < recv_images.length; i++) {
					if (!images.includes(recv_images[i])) {
						new_images.push(recv_images[i]);
						images.push(recv_images[i]);
					}
				}

				for (var i = 0; i < new_images.length; i++) {
					newImgCallback(new_images[i]);
				}

				if (!data["running"]) {
					clearInterval(interval);
				}
			}
		}
		xhttp.open("GET", "attack/list_images?pid=" + pid, true);
		xhttp.send();
	}
}

function getClassification(img_name, callback) {
	function sendRequest() {
		var xhttp = new XMLHttpRequest();
		xhttp.onreadystatechange = function() {
			if (this.readyState == 4 && this.status == 200) {
				var data = JSON.parse(xhttp.responseText);
				callback(data["remote"]);
			} else if (this.readyState == 4 && this.status == 409) {
				// The server is currently processing another request
				setTimeout(sendRequest, 1500);
			} else if (this.readyState == 4 && this.status == 503) {
				// The server is currently processing our request
				setTimeout(sendRequest, 1500);
			}
		}
		xhttp.open("GET", "model/classify?image=" + img_name + "&pid=" + pid, true);
		xhttp.send();
	}

	sendRequest();
}