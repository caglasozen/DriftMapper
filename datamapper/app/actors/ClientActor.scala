package actors

import javax.inject.Inject

import akka.actor.{Actor, ActorRef, Props}
import global.DiscretizeDateNum
import play.api.libs.json.{JsValue, Json}
import services._
import weka.core.Instances
import weka.core.converters.ArffLoader.ArffReader
import weka.core.converters.ConverterUtils.DataSource

/**
  * Created by LoongKuan on 1/07/2017.
  */

class ClientActor @Inject() (out: ActorRef, filename: String)
                            (implicit socketMessage: SocketMessage) extends Actor {

  def handleMessage(msg: JsValue) : JsValue = {
    println(msg)
    socketMessage.jsValueToMessageClass(msg) match {
      case TimelineForm(e, mt, inc, gs, dt, sl, a, ga)
        => val returnCode = InstancesReader.startTimeLineAnalysis(TimelineForm(e, mt, inc, gs, dt, sl, a, ga),
                                                                  out, filename)
        val metadata: JsValue = InstancesReader.getMetadata(filename)
        Json.obj("messageType" -> "timelineComplete", "value" -> Json.obj(
          "structure" -> metadata,
          "nInstances" -> returnCode
        ))
    }
  }

  override def preStart(): Unit = {
    println("Connected.")
  }

  def receive: Receive = {
    case msg: JsValue =>
      val response = handleMessage(msg)
      out ! response
  }

  override def postStop(): Unit = {
    println("Disconnected.")
  }
}

// Var for instance structure
object ClientActor {
  val socketMessage = new AtomicSocketMessage
  implicit var dataSource: DataSource = _
  implicit var structure: Instances= _
  implicit var discretizeDateNum: DiscretizeDateNum = _
  def props(out: ActorRef, filename: String) = Props(new ClientActor(out, filename)(socketMessage))
}

