/**
 * Created by LoongKuan on 28/06/2017.
 */
function newTimeline(metadata) {
    var data = [];
    for (var i = 0; i < metadata.attributeSubsets.length; i++) {
        data.push({x:[], y:[], type:'scatter', name:metadata.attributeSubsets[i]})
    }
    console.log("timeline-" + metadata.driftType.toString().toLowerCase());
    Plotly.newPlot('timeline-' + metadata.driftType.toString().toLowerCase(), data);
}


function updateTimeline(data) {
    var x = data.driftPoint;
    var y = data.driftMagnitudes;
    var yExtend = [];
    var xExtend = [];
    var trace = [];
    for (var i = 0; i < y.length; i++) {
        yExtend.push(y[i]);
        xExtend.push(x);
        trace.push(i);
    }
    console.log(yExtend);
    Plotly.extendTraces("timeline-" + data.driftType.toString().toLowerCase(), {x: xExtend, y:yExtend}, trace);
}