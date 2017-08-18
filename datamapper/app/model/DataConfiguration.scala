/**
  * Created by LoongKuan on 23/07/2017.
  */
package model

import play.api.libs.json.{JsPath, Reads}
/*
object DataConfiguration {
  implicit val locationReads: Reads[DataConfiguration] =
    (JsPath \ "classAttribute").read[String](DataConfiguration.apply _)
}
*/
case class DataConfiguration(classAttribute: String)

