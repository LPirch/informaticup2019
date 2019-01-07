function delete_proc(pid) {
	var xhttp = new XMLHttpRequest();
	xhttp.onreadystatechange = function() {
		if (this.readyState == 4 && this.status == 200) {
			window.location.reload(true);
		}
	}
	xhttp.open("POST", "delete_proc", true);
	xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
	xhttp.setRequestHeader("X-CSRFToken", getCookie("csrftoken"));
	xhttp.send("pid=" + pid);
}