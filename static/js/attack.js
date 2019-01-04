(function() {
	var select = document.getElementById("attack-selector");
	var options = Array.prototype.slice.call(document.querySelectorAll(".attack-options"));

	var change = function(e) {
		options.forEach(function(d) {
			if (d.id.startsWith(select.value)) {
				d.style.display = "block";
			} else {
				d.style.display = "none";
			}
		});

		var mask_input = document.querySelector('#mask_input');
		if (select.value === 'physical') {
			// 'true' value not required but can't pass only an attribute's name
			mask_input.setAttribute('required', 'true');
		} else {
			mask_input.removeAttribute('required');
		}
	}

	select.addEventListener("change", change);
	change();
})();

(function() {
	var wrapper = Array.prototype.slice.call(document.querySelectorAll(".input-file"));

	wrapper.forEach(function(input) {
		var ghost_input = input.querySelector(".input-ghost");
		var button = input.querySelector("button");
		var input_selected_file = input.querySelector(".input-selected-file");

		var onClick = function(e) {
			e.preventDefault();
			e.stopPropagation();
			ghost_input.click();
		}

		button.addEventListener("click", onClick);
		input_selected_file.addEventListener("click", onClick);

		ghost_input.addEventListener("change", function(e) {
			input_selected_file.value = ghost_input.files[0].name;
		});
	});
})();
