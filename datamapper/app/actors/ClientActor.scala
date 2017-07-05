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

class ClientActor @Inject() (out: ActorRef)
                            (implicit socketMessage: SocketMessage,
                             instancesReader: InstancesReader) extends Actor {

  def handleMessage(msg: JsValue) : JsValue = {
    println(msg)
    socketMessage.jsValueToMessageClass(msg) match {
      case LoadFile(error, file)
        => println(error)
        println(file)
        ClientActor.dataSource = instancesReader.getDataReader("../datasets/elecNormNewClean.arff")
        ClientActor.structure = ClientActor.dataSource.getStructure
        val metadata: JsValue = instancesReader.getMetadata(ClientActor.structure)
        Json.obj("messageType" -> "uploadComplete", "value" -> metadata)
      case DataConfig(error, config)
        => println(error)
        ClientActor.structure = instancesReader.setClassIndex(ClientActor.structure, config)
        println(ClientActor.structure.classIndex())
        ClientActor.discretizeDateNum = instancesReader.configureDiscretizer(ClientActor.dataSource, ClientActor.structure)
        ClientActor.structure = ClientActor.discretizeDateNum.getDiscreteStructure
        Json.obj("messageType" -> "configComplete", "value" -> instancesReader.getMetadata(ClientActor.structure))
      case TimelineForm(e, mt, inc, gs, dt, sl, a, ga)
        => val returnCode = instancesReader.startTimeLineAnalysis(TimelineForm(e, mt, inc, gs, dt, sl, a, ga),
                                                                  out,
                                                                  ClientActor.dataSource,
                                                                  ClientActor.discretizeDateNum)
        Json.obj("messageType" -> "timelineComplete", "value" -> returnCode)
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
  val instancesReader = new InstancesReader
  implicit var dataSource: DataSource = _
  implicit var structure: Instances= _
  implicit var discretizeDateNum: DiscretizeDateNum = _
  def props(out: ActorRef) = Props(new ClientActor(out)(socketMessage, instancesReader))
}

