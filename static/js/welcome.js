document.addEventListener("DOMContentLoaded", function() {
	var modal = $('#api-key-modal');
	if(modal) {
		modal.modal({show:false, backdrop: 'static', keyboad:false});
		modal.modal('show');
	}
});

function save_api_key() {
	var xhttp = new XMLHttpRequest();
	xhttp.onreadystatechange = function() {
		if (this.readyState == 4 && this.status == 200) {
			$('#api-key-modal').modal('hide');
		}

		if (this.readyState == 4 && this.status == 409) {
			console.log("huhu");
			$('#api_key').addClass('is-invalid');
			$('#no-connection').addClass('hidden');
			$('#invalid-input').removeClass('hidden');
		}

		if (this.readyState == 4 && this.status == 500) {
			$('#api_key').addClass('is-invalid');
			$('#invalid-input').addClass('hidden');
			$('#no-connection').removeClass('hidden');
		}
	}
	xhttp.open("POST", "/welcome/save_api_key", true);
	xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
	xhttp.setRequestHeader("X-CSRFToken", getCookie("csrftoken"));
	xhttp.send("api_key=" + document.querySelector('#api_key').value);
}