/* 
 * Add loading the active tab from local storage.
 */
$(document).ready(function(){
    $('ul.nav-pills li a').click(function (e) {
        $('ul.nav-pills li.active').removeClass('active');
        $(this).parent('li').addClass('active');
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