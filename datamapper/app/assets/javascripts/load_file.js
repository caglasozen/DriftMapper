/**
 * Created by LoongKuan on 1/07/2017.
 */


$("#timelineBtn").click(function(event) {
    event.preventDefault();
    ws.send(JSON.stringify({messageType: "timeline"}))
});

$("#tmpLoadFile").click(function (event) {
    event.preventDefault();
    ws.send(JSON.stringify({messageType: "load", error: "none", path: "elec"}))
});

function enableConfigForm(structure) {
    for (var i = 0; i < structure.attributes.length; i++) {
        var att = structure.attributes[i];
        var opt = document.createElement(att.toString());
        opt.value = att.toString();
        opt.innerHTML = att.toString();
        classAttribute.options[classAttribute.options.length] = new Option(att, att);
    }

    var timelineForm = document.getElementById("data-config-form");
    timelineForm.disabled = false;
}

$("#configBtn").click(function (event) {
    event.preventDefault();
    console.log("test");
    ws.send(JSON.stringify({messageType: "config", error: "none", classAttribute: $("#classAttribute").val()}))
});