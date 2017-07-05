package controllers

import java.io.{BufferedReader, FileReader}
import java.nio.file.Paths
import javax.inject._

import actors.ClientActor
import akka.actor.ActorSystem
import akka.stream.Materializer
import analyse.timeline.Chunks
import global.DriftMeasurement
import models.Model
import play.api.mvc._
import play.api.data._
import play.api.data.Forms._
import play.api.i18n.MessagesApi
import play.api.libs.json.JsValue
import play.api.libs.streams.ActorFlow
import play.api.mvc.MultipartFormData.FilePart
import play.api.mvc._
import play.core.parsers.Multipart.FileInfo
import services.TimelineForm
import weka.core.Instances
import weka.core.converters.ArffLoader.ArffReader

import scala.concurrent.ExecutionContext


case class csvFile(name: String)
/**
 * This controller creates an `Action` to handle HTTP requests to the
 * application's home page.
 */
@Singleton
class OverviewController @Inject()(cc: ControllerComponents)
                                  (implicit system: ActorSystem, mat: Materializer, ec: ExecutionContext)
  extends AbstractController(cc) with play.api.i18n.I18nSupport{

  // Use a direct reference to SLF4J
  private val logger = org.slf4j.LoggerFactory.getLogger("controllers.HomeController")

  val csvFileForm: Form[csvFile] = Form(
    mapping(
      ".csv File" -> text
    )(csvFile.apply)(csvFile.unapply)
  )

  /**
   * Create an Action to render an HTML page with a welcome message.
   * The configuration in the `routes` file means that this method
   * will be called when the application receives a `GET` request with
   * a path of `/`.
   */
  def index = Action { implicit request =>
    Ok(views.html.overview(csvFileForm))
  }

  def socket: WebSocket = WebSocket.accept[JsValue, JsValue] { request =>
    ActorFlow.actorRef { out =>
      ClientActor.props(out)
    }
  }

  /**
    *
    * @return
    */
  def upload = Action(parse.multipartFormData) { implicit request =>
    request.body.file(".csv File").map { csv =>
      val filename = csv.filename
      val contentType = csv.contentType
      csv.ref.moveTo(Paths.get(s"./tmp/csv/$filename"), replace = true)
      Ok("File uploaded")
    }.getOrElse {
      Redirect(routes.OverviewController.index).flashing(
        "error" -> "Missing file")
    }
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
