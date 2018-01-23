feather.replace();
$(function () {
    $('[data-toggle="tooltip"]').tooltip({"html": true})
});
$(function () {
    $('[data-toggle="popover"]').popover({
        "html": true,
        trigger: 'hover click'
    })
});
