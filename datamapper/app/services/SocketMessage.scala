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
    (JsPath \ "error").read[String] and
    (JsPath \ "modelType").read[String] and
    (JsPath \ "increment").read[Boolean] and
    (JsPath \ "groupSize").read[Int] and
    (JsPath \ "driftTypes").read[Array[String]] and
    (JsPath \ "subsetLength").read[Int] and
    (JsPath \ "attributes").read[Array[String]] and
    (JsPath \ "groupAttribute").read[String]
  )(TimelineForm.apply _)

  implicit val configReads: Reads[DataConfig] = (
    (JsPath \ "error").read[String] and
    (JsPath \ "classAttribute").read[String]
  )(DataConfig.apply _)

  implicit val fileReads: Reads[LoadFile] = (
    (JsPath \ "error").read[String] and
    (JsPath \ "path").read[String]
    )(LoadFile.apply _)

  override def jsValueToMessageClass(jsValue: JsValue): MessageClass = {
    (jsValue \ "messageType").as[String] match {
      case "timeline" => Json.fromJson[TimelineForm](jsValue) match {
        case JsSuccess(timeline, path) => timeline
        case e: JsError => TimelineForm(JsError.toJson(e).toString(), "", false, -1, Array(), -1, Array(), "")
      }
      case "load" => Json.fromJson[LoadFile](jsValue) match {
        case JsSuccess(load, path) => load
        case e: JsError => LoadFile(JsError.toJson(e).toString(), "")
      }
      case "config" => Json.fromJson[DataConfig](jsValue) match {
        case JsSuccess(config, path) => config
        case e: JsError => DataConfig(JsError.toJson(e).toString(), "")
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

case class TimelineForm(error:String,
                        modelType: String,
                        increment: Boolean,
                        groupSize: Int,
                        driftTypes: Array[String],
                        subsetLength: Int,
                        attributes: Array[String],
                        groupAttribute: String) extends MessageClass

case class LoadFile(error: String, file: String) extends MessageClass

case class DataConfig(error: String, classAttribute: String) extends MessageClass