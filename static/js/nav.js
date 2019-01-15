/* 
 * Add loading the active tab from local storage.
 */
$(document).ready(function(){
    $('ul.nav-pills li a').click(function (e) {
        if (! $( this ).hasClass( "disabled" ) ) {
            $('ul.nav-pills li.active').removeClass('active');
            $(this).parent('li').addClass('active');
            disable_nav();
        }
    });
    $('a[data-toggle="tab"]').on('show.bs.tab', function(e) {
        localStorage.setItem('activeTab', $(e.target).attr('href'));
    });
    set_active_tab(localStorage.getItem('activeTab'));
});

/* 
 * Set the active according to value stored in local storage.
 * @param {target} the target url (tab id)  
 */
function set_active_tab(target) {
    $('.nav-item.active, .tab-pane.active').removeClass('active');
    var activeTab = localStorage.getItem('activeTab');
    $('a[href="'+target+'"]').parent().addClass('active');
    $('div'+target).addClass('active');
}

function disable_nav() {
    $('ul.nav-pills li a').addClass('disabled');
    setTimeout(function(){
        $('ul.nav-pills li a').removeClass('disabled');
        console.log("enabling nav");
    }, 3000);
}