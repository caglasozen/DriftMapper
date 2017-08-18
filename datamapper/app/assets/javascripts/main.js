/**
 * Created by LoongKuan on 28/07/2017.
 */

$( document ).ready(function() {

    //var r = pageRoutes.controllers.OverviewController.analysisPage;
    //console.log(r.url);
    //$.ajax({url: r.url, type: r.type});
    enableAllForm()
});

function disableConfigForm() {
    $('#classAttribute').html('');
    var timelineForm = document.getElementById("data-config-form");
    timelineForm.disabled = true;
}

function enableConfigForm(structure) {
    $('#classAttribute').html('');
    for (var i = 0; i < structure.attributes.length; i++) {
        var att = structure.attributes[i];
        var opt = document.createElement(att.toString());
        opt.value = att.toString();
        opt.innerHTML = att.toString();
        classAttribute.options[classAttribute.options.length] = new Option(att, att);
    }

    var timelineForm = document.getElementById("data-config-form");
    timelineForm.disabled = false;
    $('#fileLoaded').text(structure.fileName);
}

function enableAllForm() {
    console.log("Enabling all forms");
    var r = jsRoutes.controllers.OverviewController.getDatasetStructure();
    $.ajax({url: r.url, type: r.type})
        .done(function (structure) {
            console.log(structure);
            enableConfigForm(structure);
            enableTimelineForm(structure);
            enableHeatmapForm(structure);
        });
}
