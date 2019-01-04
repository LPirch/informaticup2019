document.addEventListener("DOMContentLoaded", function() {
	document.querySelector('#modelname').addEventListener('change', function() {
		select_model(this.value);
	});
  });