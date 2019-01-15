/**
 * Initialize event handlers, enable popover (opt-in API)
 */
$(document).ready(function(){
	// init delete confirmation popover
	$('#confirm-delete').on('show.bs.modal', function(e) {
		$(this).find('.btn-ok').attr('href', $(e.relatedTarget).data('href'));
		$('.debug-url').html('Delete URL: <strong>' + $(this).find('.btn-ok').attr('href') + '</strong>');
	});
	// init warning popover when trying to upload an empty file name
	$('#input-selected-file').popover({
		container: "body",
		html: true,
		placement: 'bottom',
		content: function () {
		  return '<div class="popover-message">Please select a file here.</div>';
		}
	  });

	// this is required to trigger the popover on a different element than the one who emitted the click event (submit button)
	$(".input-file").before(
		function() {
			if ( ! $(this).prev().hasClass('input-ghost') ) {
				var element = $("<input type='file' class='input-ghost' style='visibility:hidden; height:0'>");
				element.attr("name",$(this).attr("name"));
				element.change(function(){
					element.next(element).find('input').val((element.val()).split('\\').pop());
				});
				$(this).find("button.btn-choose").click(function(){
					element.click();
				});
				$(this).find('input').css("cursor","pointer");
				$(this).find('input').mousedown(function() {
					$(this).parents('.input-file').prev().click();
					return false;
				});
				return element;
			}
		}
	);
	
	// add popover toggle logic when uploading an empty file
	$('#upload-form').submit(function (e) {
		if ($('#input-selected-file').val().trim() === "") {
			$('#input-selected-file').addClass('is-invalid');
			$('#input-selected-file').popover('toggle');
			// fade out popover after 5 seconds
			setTimeout(function(){
				$('#input-selected-file').removeClass('is-invalid');
				$('#input-selected-file').popover('toggle');
			}, 5000);
			return false;
		}
	});

	$("a.fa-trash-alt").each(function(){
		$(this).confirmation({
			rootSelector: '[data-toggle=confirmation]',
		});
		$(this).click(function() {
			var filename = $( this ).parent().siblings(".model-filename").text();
			delete_model(filename);
		});
	});
});

 /*
 * Send POST request for deleting a model file. 
 * @param {file} file name of the file to delete
 */
function delete_model(file) {
	var xhttp = new XMLHttpRequest();
	xhttp.onreadystatechange = function() {
		if (this.readyState == 4 && this.status == 200) {
		var tds = document.querySelectorAll('.model-filename');
			for (var i = 0; i < tds.length; i++) {
				if(tds[i].innerText === file) {
					tds[i].parentNode.remove();
					break;
				}
			}
		}
	}

	xhttp.open("POST", "/models/overview/deletemodel", true);
	xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
	xhttp.setRequestHeader("X-CSRFToken", getCookie("csrftoken"));
	xhttp.send("filename=" + file);
};

function abort(pid) {
	var xhttp = new XMLHttpRequest();
	xhttp.onreadystatechange = function() {
		if (this.readyState == 4 && this.status == 200) {
			window.location.assign('/models/overview.html');
		}
	}

	xhttp.open("POST", "/models/overview/abort_training", true);
	xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
	xhttp.setRequestHeader("X-CSRFToken", getCookie("csrftoken"));
	xhttp.send("pid=" + pid);
}