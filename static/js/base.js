document.addEventListener("DOMContentLoaded", function() {
	setTimeout(function(){
		// chrome sometimes skips this animation if we don't wait 
		document.querySelector('.content').classList.add('in');
		// chrome fails to set the textArea resizer color on page load
		document.styleSheets[0].insertRule('.container textarea::-webkit-resizer {background: black;}', 0)
	}, 1);
});

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