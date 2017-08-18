/**
 * Created by LoongKuan on 2/07/2017.
 */


// TODO: web socket to request instance structure, disable controls if return none

// TODO: If has instances, then enable controlls, do value restriction and validification

function enableTimelineForm(instancesStructure) {
    var timelineForm = document.getElementById("timeline-form");
    timelineForm.disabled = false;
    document.getElementById("subsetLength").min = 1;
    document.getElementById("subsetLength").max = instancesStructure.attributes.length;
    document.getElementById("groupSize").min = 1;
    $('#groupAttribute').html('');
    $('#attributes').html('');
    var attSelect = document.getElementById("attributes");
    var groupSelect = document.getElementById("groupAttribute");
    for (var i = 0; i < instancesStructure.attributes.length; i++) {
        var att = instancesStructure.attributes[i];
        var opt = document.createElement(att.toString());
        opt.value = att.toString();
        opt.innerHTML = att.toString();
        attSelect.options[attSelect.options.length] = new Option(att, att);
        groupSelect.options[groupSelect.options.length] = new Option(att, att);
    }
    document.getElementById("modelType").value = "Chunks";
}

function disableTimelineForm() {
    var timelineForm = document.getElementById("timeline-form");
    timelineForm.disabled = true;
    $('#groupAttribute').html('');
    $('#attributes').html('');
}

$("#modelType").change(function (event) {
    event.preventDefault();
    switch (document.getElementById("modelType").value){
        case "Chunks":
            document.getElementById("groupSizeLabel").innerHTML = "Sequence Keys per Group:";
            document.getElementById("groupAttribute").disabled = false;
            break;
        case "Windows":
            document.getElementById("groupSizeLabel").innerHTML = "Number of Instances per Group:";
            document.getElementById("groupAttribute").disabled = true;
            break;
    }
});

function getDriftTypes() {
    var driftTypes = [];
    if (document.getElementById('typeClass').checked) driftTypes.push("Class");
    else Plotly.purge("timeline-class");
    if (document.getElementById('typeCovariate').checked) driftTypes.push("Covariate");
    else Plotly.purge("timeline-covariate");
    if (document.getElementById('typePosterior').checked) driftTypes.push("Posterior");
    else Plotly.purge("timeline-posterior");
    if (document.getElementById('typeLikelihood').checked) driftTypes.push("Likelihood");
    else Plotly.purge("timeline-likelihood");
    if (document.getElementById('typeJoint').checked) driftTypes.push("Joint");
    else Plotly.purge("timeline-joint");
    console.log(driftTypes);
    return driftTypes;
}

$("#timelineBtn").click(function (event) {
    event.preventDefault();
    var classAttribute = document.getElementById("classAttribute").value;
    var modelType = document.getElementById("modelType").value;
    var increment = document.getElementById("increment").checked;
    var groupSize = +document.getElementById("groupSize").value;
    var driftTypes = getDriftTypes();
    var subsetLength = +document.getElementById("subsetLength").value;
    var attributes = $('#attributes').val();
    var groupAttribute = document.getElementById("groupAttribute").value;


    ws = new WebSocket($("body").data("ws-url"));
    setTimeout(function () {
        // Send data
        ws.send(JSON.stringify({
            messageType: "timeline",
            classAttribute: classAttribute,
            modelType: modelType,
            increment: increment,
            groupSize: groupSize,
            driftTypes: driftTypes,
            subsetLength: subsetLength,
            attributes: attributes,
            groupAttribute: groupAttribute
        }));
    }, 1000);

    ws.onmessage = function (event) {
    var message = JSON.parse(event.data);
    console.log(message);
    switch (message.messageType) {
        case "timelineHeader":
            newTimeline(message.value);
            break;
        case "timelineUpdate":
            updateTimeline(message.value);
            break;
        case "timelineComplete":
            //enableAnalysisForm(message.value);
            ws.close();
            break;
        case "analysisComplete":
            createHeatmap(message.value)
    }
};
});
