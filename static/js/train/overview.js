function getModelInfo(modelname) {
	// remove file ending
	modelname = modelname.split(".").slice(0, -1).join(".")
	loading_started();

	var xhttp = new XMLHttpRequest();
	xhttp.open("GET", "/train/start_model_info?modelname="+modelname);
	xhttp.send();

	var fetchInterval = setInterval(function() {
		if(modelname) {
			var csrftoken = getCookie('csrftoken');
			var xhttp = new XMLHttpRequest();
			xhttp.open("POST", "/train/model_info", true);
			xhttp.onreadystatechange = function() {
				if (this.readyState == 4 && this.status == 200) {
					var modelInfo = JSON.parse(xhttp.responseText);
					
					var parent = document.querySelector('#modal-content');

					var table = document.createElement('table');
					// somehow, bootstrap table-striped doesn't work in modal
					table.classList.add('table', 'striped');
					
					var thead = document.createElement('thead');
					var tr = document.createElement('tr');
					['Layer', 'Input Shape', 'Output Shape'].map(function(x){
						var th = document.createElement('th');
						th.innerText = x;
						tr.appendChild(th);
					});
					thead.appendChild(tr);

					var tbody = document.createElement('tbody');
					modelInfo.map(function(layer) {
						var tr = document.createElement('tr');
						
						var name = document.createElement('td');
						name.innerText = layer.name;
						tr.appendChild(name);
						
						var input_shape = document.createElement('td');
						input_shape.innerText = layer.input_shape;
						tr.appendChild(input_shape);

						var output_shape = document.createElement('td');
						output_shape.innerText = layer.output_shape;
						tr.appendChild(output_shape);

						tbody.appendChild(tr);
					});

					table.appendChild(thead);
					table.appendChild(tbody);
					parent.appendChild(table);

					loading_finished();
					clearInterval(fetchInterval);
				}
			}
			xhttp.setRequestHeader('Content-Type', 'application/json');
			xhttp.setRequestHeader("X-CSRFToken", csrftoken);
			xhttp.send(JSON.stringify({'modelname': modelname}));
		}
	}, 2000)
}

function getCookie(cname) {
	var name = cname + "=";
	var decodedCookie = decodeURIComponent(document.cookie);
	var ca = decodedCookie.split(';');
	for(var i = 0; i <ca.length; i++) {
		var c = ca[i];
		while (c.charAt(0) == ' ') {
			c = c.substring(1);
		}
		if (c.indexOf(name) == 0) {
			return c.substring(name.length, c.length);
		}
	}
	return "";
}

function loading_started() {
	document.querySelector('#modal-content').classList.remove('active');
	document.querySelector('#modal-content').parentElement.classList.add('flex-center');
	document.querySelector('#loading').classList.add('active');
}

function loading_finished() {
	document.querySelector('#loading').classList.remove('active');
	document.querySelector('#modal-content').parentElement.classList.remove('flex-center');
	document.querySelector('#modal-content').classList.add('active');
}