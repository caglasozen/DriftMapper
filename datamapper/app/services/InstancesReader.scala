package services

import java.io.{BufferedReader, FileReader}
import java.util

import akka.actor.{ActorRef, Status}
import akka.util.ByteString
import analyse.timeline.{Chunks, TimelineAnalysis, Windows}
import global.{DiscretizeDateNum, DriftMeasurement}
import model.TimelineConfig
import models.Model
import models.frequency.FrequencyMaps
import play.api.libs.json.{JsValue, Json}
import result.timeline.TimelineListener
import weka.core.{Instance, Instances}
import weka.core.converters.ArffLoader.ArffReader
import weka.core.converters.ConverterUtils.DataSource

import scala.collection.immutable.Range
import scala.concurrent.{ExecutionContext, Future}


/**
  * Created by LoongKuan on 1/07/2017.
  */

object InstancesReader {

  def getDataReader(filepath: String): DataSource = {
    new DataSource(filepath)
  }

  def printStructure(filepath:String): Unit = {
    val dataSource: DataSource = this.getDataReader(filepath)
    println(dataSource.getStructure)
  }

  def getMetadata(filepath: String): JsValue = {
    val instanceStructure = this.getDataReader(filepath).getDataSet
    val attributes: IndexedSeq[String] =
      for (n <- Range(0, instanceStructure.numAttributes())) yield instanceStructure.attribute(n).name()
    val nInstances: Int = instanceStructure.numInstances()
    val json = Json.obj(
      "fileName" -> filepath.split("-").last,
      "attributes" -> attributes,
      "nInstances" -> nInstances
    )
    println(json)
    json
  }

  def setClassIndex(structure: Instances, classAtt: String): Instances = {
    val attributes: IndexedSeq[String] = for (n <- Range(0, structure.numAttributes())) yield structure.attribute(n).name()
    val classIndex: Int = attributes.indexOf(classAtt)
    structure.setClassIndex(classIndex)
    structure
  }

  def configureDiscretizer(dataSource: DataSource, structure: Instances): DiscretizeDateNum = {
    new DiscretizeDateNum(dataSource, structure)
  }

  private def createListener(driftMeasurement: DriftMeasurement,
                             messageAccumulator: MessageAccumulator): TimelineListener = {
    new TimelineListener {
      override def updateMetaData(attributeSubsets: Array[String]): Unit = {
        println("updating metadata")
        messageAccumulator.getAttachedActor ! ByteString.fromString(
          Json.obj(
            "messageType" -> "timelineHeader",
            "value" -> Json.obj(
              "driftType" -> driftMeasurement.toString,
              "attributeSubsets" -> attributeSubsets)).toString() + ";"
        )
      }

      override def returnDriftPointMagnitude(driftPoint: Int, driftMagnitude: Array[Double]): Unit = {
        messageAccumulator.addBacklog(driftPoint, driftMagnitude, driftMeasurement.toString)
      }

      override def getMeasurementType: DriftMeasurement = driftMeasurement
    }
  }

  def startTimeLineAnalysis(config: TimelineConfig, out: ActorRef, filename: String)(implicit ec: ExecutionContext): Future[Int] = Future {
    val dataSource = this.getDataReader(filename)

    val tmpStructure = setClassIndex(dataSource.getStructure, config.classAttribute)
    val discretizeDateNum = this.configureDiscretizer(dataSource, tmpStructure)

    dataSource.hasMoreElements(dataSource.getStructure())
    dataSource.reset()

    val structure = discretizeDateNum.getDiscreteStructure
    val attributes: IndexedSeq[String] = for (n <- Range(0, structure.numAttributes())) yield structure.attribute(n).name()
    val attributeIndices: IndexedSeq[Int] = for (att <- config.attributes) yield attributes.indexOf(att, 0)
    val chunkAttIndex: Int = attributes.indexOf(config.groupAttribute)

    //TODO: Only use frequecy map for model, ergo removing need to send moodel reference?
    val model: FrequencyMaps  = new FrequencyMaps(structure, config.subsetLength, attributeIndices.toArray)
    val analysis = config.modelType match {
      case "Chunks" => new Chunks(chunkAttIndex, config.groupSize, config.increment, model)
      case "Windows" => new Windows(config.groupSize, config.increment, model)
    }

    val accumulators: IndexedSeq[MessageAccumulator] =
      for (driftType <- config.driftTypes)
        yield new MessageAccumulator(out, 500, DriftMeasurement.valueOf(driftType.toUpperCase))

    val timelineListeners: IndexedSeq[TimelineListener] =
      for (accumulator <- accumulators)
        yield createListener(accumulator.getDriftType, accumulator)

    dataSource.reset()
    for (listener <- timelineListeners) yield analysis.addListener(listener)
    analysis.updateListenerMetadata()
    val returnCode = runTimelineAnalysis(analysis, dataSource, structure, discretizeDateNum, 0)
    //TODO: Add listener post
    for (accumulator <- accumulators) yield accumulator.forwardAll()
    for (accumulator <- accumulators) yield accumulator.getAttachedActor ! Status.Success(())
    returnCode
  }

  private def runTimelineAnalysis(analysis: TimelineAnalysis,
                                  dataSource: DataSource,
                                  instanceStructure: Instances,
                                  discretizeDateNum: DiscretizeDateNum,
                                  nInstancesAcc: Int): Int = {
    val inst: Instance = dataSource.nextElement(dataSource.getStructure)
    inst match {
      case null =>
        dataSource.reset()
        nInstancesAcc
      case _ =>
        val discreteInst: Instance = discretizeDateNum.discretizeInstance(inst)
        analysis.addInstance(discreteInst)
        runTimelineAnalysis(analysis, dataSource, instanceStructure, discretizeDateNum, nInstancesAcc+1)
    }
  }
}