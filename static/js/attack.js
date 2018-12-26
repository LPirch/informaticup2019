(function() {
	var select = document.getElementById("attack-selector");
	var options = Array.prototype.slice.call(document.querySelectorAll(".attack-options"));

	var change = function(e) {
		for (var i = 0; i < options.length; i++) {
			if (options[i].id.startsWith(select.value)) {
				options[i].style.display = "block";
			} else {
				options[i].style.display = "none";
			}
		}
	}

	select.addEventListener("change", change);
	change();
})();