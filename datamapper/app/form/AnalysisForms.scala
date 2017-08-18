/**
  * Created by LoongKuan on 7/07/2017.
  */


package form

import play.api.data._
import play.api.data.Forms._

import model._

object AnalysisForms {
  def fileUploadInfoForm = Form(
    mapping(
      "resumableChunkNumber" -> number,
      "resumableChunkSize" -> number,
      "resumableTotalSize" -> number,
      "resumableIdentifier" -> nonEmptyText,
      "resumableFilename" -> nonEmptyText
    )(FileUploadInfo.apply)(FileUploadInfo.unapply)
  )

}