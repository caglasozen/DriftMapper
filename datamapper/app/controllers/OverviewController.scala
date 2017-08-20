package controllers

import java.io.{BufferedReader, ByteArrayOutputStream, FileReader}
import java.io.File
import java.nio.file.{Files, Paths}
import java.util.concurrent.Executors
import javax.inject._

import akka.actor.{ActorRef, ActorSystem}
import akka.stream.{Materializer, OverflowStrategy}
import akka.stream.scaladsl.{Sink, Source}
import akka.util.ByteString
import form.AnalysisForms
import model.TimelineConfig
import model.heatmap.{HeatmapConfig, WindowRange}
import play.api._
import play.api.data.Form
import play.api.i18n.MessagesApi
import play.api.libs.json._
import play.api.libs.streams.ActorFlow
import play.api.mvc._
import play.api.routing._
import play.api.libs.functional.syntax._
import services.{FileUploadService, HeatmapGenerator, InstancesReader}
import weka.core.Instances
import weka.core.converters.ConverterUtils.DataSource

import scala.collection.mutable
import scala.concurrent.ExecutionContext


/**
 * This controller creates an `Action` to handle HTTP requests to the
 * application's home page.
 */
@Singleton
class OverviewController @Inject()(cc: ControllerComponents, messagesApi: MessagesApi)
                                  (implicit system: ActorSystem, mat: Materializer, ec: ExecutionContext)
  extends AbstractController(cc) with play.api.i18n.I18nSupport {

  // Use a direct reference to SLF4J
  private val logger = org.slf4j.LoggerFactory.getLogger("controllers.HomeController")
  private var fileUploaded: mutable.Map[String, String] = mutable.Map[String, String]()

  implicit val windowRangeReads: Reads[WindowRange] = (
    (JsPath \ "start").read[Int] and
      (JsPath \ "end").read[Int]
  )(WindowRange.apply _)

  implicit val heatmapConfigReads: Reads[HeatmapConfig] = (
    (JsPath \ "driftTypes").read[Array[String]] and
      (JsPath \ "attributes").read[Array[String]] and
      (JsPath \ "classAttribute").read[String] and
      (JsPath \ "window1").read[WindowRange] and
      (JsPath \ "window2").read[WindowRange]
  )(HeatmapConfig.apply _)

  implicit val timelineConfigReads: Reads[TimelineConfig] = (
      (JsPath \ "classAttribute").read[String] and
      (JsPath \ "modelType").read[String] and
      (JsPath \ "increment").read[Boolean] and
      (JsPath \ "groupSize").read[Int] and
      (JsPath \ "driftTypes").read[Array[String]] and
      (JsPath \ "subsetLength").read[Int] and
      (JsPath \ "attributes").read[Array[String]] and
      (JsPath \ "groupAttribute").read[String]
    )(TimelineConfig.apply _)

  def jsRoutes = Action { implicit request: Request[AnyContent] =>
    Ok(
      JavaScriptReverseRouter("jsRoutes")(
        routes.javascript.OverviewController.getDatasetStructure,
        routes.javascript.OverviewController.getNewSession,
        routes.javascript.OverviewController.analysisPage,
        routes.javascript.OverviewController.getTimeline,
        routes.javascript.OverviewController.getHeatmap
      )
    ).as("text/javascript")
  }

  def analysisPage = Action { implicit request: Request[AnyContent] =>
    println(request.session)
    Ok(views.html.analysis())
  }

  def aboutPage = Action { implicit request: Request[AnyContent] =>
    println(request.session)
    Ok(views.html.about())
  }

  def getDatasetStructure = Action { implicit request: Request[AnyContent] =>
    request.session.get("data").map { data =>
      Ok(InstancesReader.getMetadata(request.session("data")))
    }.getOrElse {
      Unauthorized("No data file uploaded")
    }
  }

  def getNewSession = Action(parse.json) { implicit request: Request[JsValue] =>
    val fileId = (request.body \ "id").as[String]
    if (this.fileUploaded.contains(fileId)) {
      Ok("Session Updated").withSession(
        "data" -> this.fileUploaded(fileId)
      )
    }
    else {
      BadRequest("File Not Uploaded")
    }
  }

  def getTimeline = Action(parse.json) { request: Request[JsValue] =>
    println(request)
    val timelineRequest = request.body.validate[TimelineConfig]
    timelineRequest.fold(
      errors => {
        println("Error getting timeline")
        BadRequest(Json.obj("status" ->"KO", "message" -> JsError.toJson(errors)))
      },
      config => {
        println("Getting timeline")
        val filename = request.session("data")
        val pool = Executors.newCachedThreadPool()
        implicit val ec = ExecutionContext.fromExecutorService(pool)
        val source: Source[ByteString, _] = Source
          .actorRef[ByteString](Int.MaxValue, OverflowStrategy.fail)
          .mapMaterializedValue(ref => InstancesReader.startTimeLineAnalysis(config, ref, filename))
        Ok.chunked(source)
      }
    )
  }

  def getHeatmap = Action(parse.json) { implicit request: Request[JsValue]=>
    println(request)
    val heatmapRequest = request.body.validate[HeatmapConfig]
    heatmapRequest.fold(
      errors => {
        BadRequest(Json.obj("status" ->"KO", "message" -> JsError.toJson(errors)))
      },
      config => {
        var filename = request.session("data")
        var res = Json.toJson(HeatmapGenerator.runAnalysis(filename, config))
        println(res)
        Ok(res)
      }
    )
  }

  val fileUploadService = new FileUploadService("./tmp/arff/")

  def upload = Action(parse.multipartFormData) { implicit request =>
    AnalysisForms.fileUploadInfoForm.bindFromRequest.fold(
      formWithErrors => {
        println("error: " + formWithErrors.errors.mkString("\n"))
        BadRequest(formWithErrors.errors.mkString("\n"))
      },
      fileUploadInfo => {
        request.body.file("file") match {
          case None => BadRequest("No File")
          case Some(file) =>
            val bytes = Files.readAllBytes(file.ref.path)
            fileUploadService.savePartialFile(bytes, fileUploadInfo)
            if (fileUploadService.isLast(fileUploadInfo)) {
              val filename = "./tmp/arff/" + fileUploadInfo.resumableIdentifier + "-" + fileUploadInfo.resumableFilename
              this.fileUploaded += (fileUploadInfo.resumableIdentifier -> filename)
              Ok.withSession(
                "data" -> filename
              )
            } else {
              Ok
            }
        }
      }
    )
  }


  private def discretizeData(instances: Instances) ={
    //TODO
  }

  /*
  private def createModel(formData: TimelineForm): Model = {
    formData.modelType match {
      case "chunks" => Chunks()
    }
  }
  */

}
