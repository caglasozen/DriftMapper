/**
  * Created by LoongKuan on 1/07/2017.
  */

ws = new WebSocket($("body").data("ws-url"));
ws.onmessage = function (event) {
    var message = JSON.parse(event.data);
    console.log(message);
    switch (message.messageType) {
        case "uploadComplete":
            enableConfigForm(message.value);
            break;
        case "configComplete":
            enableTimelineForm(message.value);
            break;
        case "timelineHeader":
            newTimeline(message.value);
            break;
        case "timelineUpdate":
            updateTimeline(message.value);
            break;
    }
};