/**
 * Created by LoongKuan on 1/07/2017.
 */

var resumable = new Resumable({
    target:'/upload',
    maxFiles: 1,
    //fileType: "arff",
    query: {}
});

var retries = 0;

// Manage buttons
resumable.assignBrowse(document.getElementById('browse-file'));
document.getElementById('uploadFile').onclick = function () {
    resumable.upload();
    retries = 0;
};
document.getElementById('pauseUpload').onclick = function () {
    resumable.pause();
};
document.getElementById('cancelUpload').onclick = function () {
    resumable.cancel();
    retries = 0;
};

resumable.on('fileadded', function (file) {
    console.log(file.fileName);
    document.getElementById('browse-label').innerHTML = file.fileName;
    disableConfigForm();
    disableTimelineForm();
    disableHeatmapForm();
    $('#file-progress')
        .css('width', 0 + '%')
        .attr('aria-valuenow', 0)
        .text(0 + '%');
});

resumable.on('fileProgress', function (file) {
    var percentage = file.progress(false) * 100;
    $('#file-progress')
        .css('width', percentage + '%')
        .attr('aria-valuenow', percentage)
        .text(percentage + '%');
});

resumable.on('fileSuccess', function (file, message) {
    console.debug(file);
    console.debug(message);
    $('#file-progress').text("Upload Complete!");
    $('#fileLoaded').text(file.fileName);
    enableAllForm();
    //ws.send(JSON.stringify({messageType: "load", error: "none", path: file.uniqueIdentifier + "-" + file.relativePath}))
});

resumable.on('cancel', function (file) {
    document.getElementById('uploadErrorMsg').textContent = "Upload Canceled";
});

resumable.on('error', function (message, file) {
    document.getElementById('uploadErrorMsg').textContent = message;
});

resumable.on('catchAll', function (eventX) {
    console.log(eventX);
});


$("#configBtn").click(function (event) {
    event.preventDefault();
    var message = JSON.stringify({classAttribute: $("#classAttribute").val()});
    var route = jsRoutes.controllers.OverviewController.configureDataset();
    $.ajax({url: route.url, type: route.type, data: message});
    //ws.send(JSON.stringify({messageType: "config", error: "none", classAttribute: $("#classAttribute").val()}))
});
