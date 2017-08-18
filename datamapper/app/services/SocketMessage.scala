package services

import global.DriftMeasurement
import javax.inject._

import play.api.libs.json._
import play.api.libs.json.Reads._ // Custom validation helpers
import play.api.libs.functional.syntax._
/**
  * Created by LoongKuan on 1/07/2017.
  *
  * Handle the possible types of messages received from a socket Connection
  */


abstract class MessageClass

trait SocketMessage {
  def jsValueToMessageClass(jsValue: JsValue): MessageClass
  //def messageClassToJsValue(messageClass: MessageClass): JsValue
}

@Singleton
class AtomicSocketMessage extends SocketMessage {

  implicit val timelineReads: Reads[TimelineForm] = (
    (JsPath \ "classAttribute").read[String] and
    (JsPath \ "modelType").read[String] and
    (JsPath \ "increment").read[Boolean] and
    (JsPath \ "groupSize").read[Int] and
    (JsPath \ "driftTypes").read[Array[String]] and
    (JsPath \ "subsetLength").read[Int] and
    (JsPath \ "attributes").read[Array[String]] and
    (JsPath \ "groupAttribute").read[String]
  )(TimelineForm.apply _)

  implicit val analysisReads: Reads[AnalysisForm] = (
    (JsPath \ "error").read[String] and
      (JsPath \ "driftTypes").read[Array[String]] and
      (JsPath \ "attributes").read[Array[String]] and
      (JsPath \ "window1").read[Array[Int]] and
      (JsPath \ "window2").read[Array[Int]]
    )(AnalysisForm.apply _)

  override def jsValueToMessageClass(jsValue: JsValue): MessageClass = {
    (jsValue \ "messageType").as[String] match {
      case "timeline" => Json.fromJson[TimelineForm](jsValue) match {
        case JsSuccess(timeline, path) => timeline
        case e: JsError => TimelineForm(JsError.toJson(e).toString(), "", false, -1, Array(), -1, Array(), "")
      }
      case "analysis" => Json.fromJson[AnalysisForm](jsValue) match {
        case JsSuccess(analysis, path) => analysis
        case e: JsError => AnalysisForm(JsError.toJson(e).toString(), Array(), Array(), Array(), Array())
      }
    }
  }
/*
  override def messageClassToJsValue(messageClass: MessageClass): JsValue = {
    messageClass match {
      case timeline: TimelineForm => Json.toJson(timeline)
      case load: LoadFile => Json.toJson(load)
    }
  }
  */
}

case class TimelineForm(classAttribute: String,
                        modelType: String,
                        increment: Boolean,
                        groupSize: Int,
                        driftTypes: Array[String],
                        subsetLength: Int,
                        attributes: Array[String],
                        groupAttribute: String) extends MessageClass

case class LoadFile(error: String, file: String) extends MessageClass

case class DataConfig(error: String, classAttribute: String) extends MessageClass

case class AnalysisForm(error:String,
                        driftTypes: Array[String],
                        attributes: Array[String],
                        window1: Array[Int],
                        window2: Array[Int]) extends MessageClass
