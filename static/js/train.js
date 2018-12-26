/**
 * Initialize event handlers, enable popover (opt-in API)
 */
$(document).ready(function(){
    // fetch available models on page load
    reload_models();
    // reload button on the top right of the table
    $("#reload_models_button").click(reload_models);
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
});

/* 
 * reload model table according to data fetched from django 
 */
function reload_models() {
    $.ajax({
        'url': "/train/models/reloadmodel",
        'type':"GET",
        'success': function(data, status, xhr){

            $('#models-tbody').html("");
            for (var i = 0; i < data.length; i++) {
                var model = data[i];
                var select_icon = '';
                if(model.selected) {
                    select_icon = '<i class="fa fa-check"></i>';
                }

                // generate row content
                $('#models-tbody').append('<tr class="table-active"> \
                    <td scope="row">'+(i+1)+'</td> \
                    <td class="table-selected">'+select_icon+'</td> \
                    <td class="model-filename">'+model.name+'</td> \
                    <td>'+model.modified+' </td>\
                    <td><a class="fas fa-trash-alt" data-toggle="confirmation" data-placement="right"\
                    data-btn-ok-label="Proceed" data-btn-ok-class="btn-success"\
                    data-btn-cancel-label="Cancel" data-btn-cancel-class="btn-danger"\
                    data-title="Are you sure?" data-content="This cannot be undone."></a></td>\
                    </tr>');
            }

            // show info text if no data has been passed
            if(data.length == 0) {
                $('#empty-table-info').removeClass('invisible');
                $('#model-table').addClass('invisible');
            } else {
                $('#empty-table-info').addClass('invisible');
                $('#model-table').removeClass('invisible');
            }

            $("a.fa-trash-alt").each(function(){
                $(this).confirmation({
                    rootSelector: '[data-toggle=confirmation]',
                });
                $(this).click(function() {
                    var filename = $( this ).parent().siblings(".model-filename").text();
                    delete_model(filename);
                });
                
            });
        }
    });

    // re-add click handlers for generated rows
    $("#model-table").unbind("click");
    $("#model-table").on('click', 'tbody tr', function(){
        select_model($(this).children('.model-filename').text());
    });
    return false;
 } 

 /*
 * Send GET request for deleting a model file. 
 * @param {file} file name of the file to delete
 */
function delete_model(file) {
    $.ajax({
        url: "/train/models/deletemodel",
        type: "GET",
        data: {'filename': file},
        success: function(result) {
            reload_models();
        }
    });
    return false;
};

/* 
 * Send GET request to update the django session's selected model. 
 * @param {file} file name of the selected model
 */
function select_model(file) {
    $.ajax({
        url: "/train/models/selectmodel",
        type: "GET",
        data: {'filename': file},
        success: function(result) {
            reload_models();
        }
    });
    return false;
};