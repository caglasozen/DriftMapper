/**
 * Created by LoongKuan on 2/07/2017.
 */


// TODO: web socket to request instance structure, disable controls if return none

// TODO: If has instances, then enable controlls, do value restriction and validification

function enableHeatmapForm(structure) {
    console.log("Enable Heatmap Form");
    var analysisForm = document.getElementById("analysis-form");
    analysisForm.disabled = false;

    document.getElementById("window1-from").min = 0;
    document.getElementById("window1-to").min = 1;
    document.getElementById("window2-from").min = 2;
    document.getElementById("window2-to").min = 3;

    document.getElementById("window1-from").max = structure.nInstances - 4;
    document.getElementById("window1-to").max = structure.nInstances - 3;
    document.getElementById("window2-from").max = structure.nInstances - 2;
    document.getElementById("window2-to").max = structure.nInstances - 1;

    var attSelect = document.getElementById("anaAttributes");
    $('#anaAttributes').empty();
    for (var i = 0; i < structure.attributes.length; i++) {
        var att = structure.attributes[i];
        var opt = document.createElement(att.toString().replace(/[^a-zA-Z ]/g, ""));
        opt.value = att.toString();
        opt.innerHTML = att.toString();
        attSelect.options[attSelect.options.length] = new Option(att, att);
    }
    document.getElementById("modelType").value = "Chunks";
}

function disableHeatmapForm() {
    var analysisForm = document.getElementById("analysis-form");
    analysisForm.disabled = true;
    $('#anaAttributes').html('');
}

function getAnaDriftTypes() {
    var driftTypes = [];
    if (document.getElementById('anaCovariate').checked) driftTypes.push("Covariate");
    else Plotly.purge("heatmap-covariate");
    if (document.getElementById('anaPosterior').checked) driftTypes.push("Posterior");
    else Plotly.purge("heatmap-posterior");
    if (document.getElementById('anaLikelihood').checked) driftTypes.push("Likelihood");
    else Plotly.purge("heatmap-likelihood");
    if (document.getElementById('anaJoint').checked) driftTypes.push("Joint");
    else Plotly.purge("heatmap-joint");
    console.log(driftTypes);
    return driftTypes;
}

$("#analysisBtn").click(function (event) {
    event.preventDefault();
    var driftTypes = getAnaDriftTypes();
    var attributes = $('#anaAttributes').val();
    var classAttributes = $('#classAttribute').val();
    var windowBound1 = {"start": +$('#window1-from').val(), "end": +$('#window1-to').val()};
    var windowBound2 = {"start": +$('#window2-from').val(), "end": +$('#window2-to').val()};

    // Send data
    var message = JSON.stringify({
        driftTypes: driftTypes,
        attributes: attributes,
        classAttribute: classAttributes,
        window1: windowBound1,
        window2: windowBound2
    });
    console.log(message);

    var route = jsRoutes.controllers.OverviewController.getHeatmap();
    $.ajax({url: route.url, type: route.type, data: message, contentType: 'text/json'})
        .done(function(body) {
            var res = body;
            for (var i = 0; i < res.length; i++) {
                createHeatmap(res[i]);
            }
        });
});
