function delete_proc(pid) {
	var xhttp = new XMLHttpRequest();
	xhttp.onreadystatechange = function() {
		if (this.readyState == 4 && this.status == 200) {
			window.location.reload(true);
		}
	}
	xhttp.open('GET', 'delete_proc?pid='+pid, true);
	xhttp.send();
}