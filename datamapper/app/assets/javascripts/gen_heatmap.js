/**
 * Created by LoongKuan on 28/06/2017.
 */

var colorscaleValue = [
    [0, '#00ff00'],
    [1, '#ff0000']
];

function createHeatmap(value) {
    var data = [
        {
            z: value.data,
            x: value.attributes,
            y: value.attributes,
            colorscale: colorscaleValue,
            name: "drift magnitude",
            type: 'heatmap'
        }
    ];
    var layout = {
        title: value.type.toLowerCase() + " (Marginal length of 2)",
        width: 600,
        height: 500
    };
    Plotly.newPlot('heatmap-' + value.type.toLowerCase(), data, layout);
}