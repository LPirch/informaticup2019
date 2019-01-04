document.addEventListener("DOMContentLoaded", function() {
	document.querySelector('.content').classList.add('in');
});

/* 
 * Send GET request to update the django session's selected model. 
 * @param {file} 		file name of the selected model
 * @param {callback} 	optional callback which is called on receiving HTTP 200
 */
function select_model(file, callback) {
	var xhttp = new XMLHttpRequest();
	if (callback) {
		xhttp.onreadystatechange = function() {
			if (this.readyState == 4 && this.status == 200) {
				callback(file);
			}
		}
	}
	xhttp.open("GET", "/selectmodel?filename="+file, true);
	xhttp.send();

	return false;
};

/*
 * Retrieves the cookie value for a given cookie name.
  * @param(cname) cookie name (string)
 */
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