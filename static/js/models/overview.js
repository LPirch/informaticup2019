function getModelInfo(modelname) {
	loading_started();

	// remove file ending
	modelname = modelname.split(".").slice(0, -1).join(".")

	if (modelname) {
		sendRequest();
	}

	function sendRequest() {
		var xhttp = new XMLHttpRequest();
		xhttp.open("GET", "/models/model_info?modelname="+modelname, true);
		xhttp.onreadystatechange = function() {
			if (this.readyState == 4 && this.status == 200) {
				var data = JSON.parse(xhttp.responseText);
				var modelInfo = data.modelInfo;
				
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
			} else if (this.readyState == 4 && this.status == 409) {
				// The server is currently processing another request
				setTimeout(sendRequest, 1500);
			} else if (this.readyState == 4 && this.status == 503) {
				// The server is currently processing our request
				setTimeout(sendRequest, 1500);
			}
		}
		xhttp.setRequestHeader("X-CSRFToken", getCookie('csrftoken'));
		xhttp.send();
	}
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