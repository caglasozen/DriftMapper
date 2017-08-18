package services

import java.util

import akka.actor.ActorRef
import analyse.StaticData
import global.{DiscretizeDateNum, DriftMeasurement}
import model.heatmap.{HeatmapConfig, WindowRange}
import play.api.libs.json.{JsValue, Json}
import result.batch.ExperimentResult
import weka.core.{Instance, Instances}
import weka.core.converters.ConverterUtils.DataSource

/**
  * Created by LoongKuan on 6/07/2017.
  */

object HeatmapGenerator {

  def runAnalysis(filename: String, config: HeatmapConfig): Array[JsValue] = {
    val dataSource = InstancesReader.getDataReader(filename)
    val tmpStructure = InstancesReader.setClassIndex(dataSource.getStructure, config.classAttribute)
    val discretizer = InstancesReader.configureDiscretizer(dataSource, tmpStructure)

    val allAttributes: IndexedSeq[String] =
      for (n <- Range(0, discretizer.getDiscreteStructure.numAttributes()))
        yield discretizer.getDiscreteStructure.attribute(n).name()
    val attributeIndices: Array[Int] = for (att <- config.attributes) yield allAttributes.indexOf(att, 0)
    for (driftType <- config.driftTypes) yield Json.obj(
      "type" -> driftType.toString,
      "attributes" -> config.attributes,
      "data" -> getMapData(dataSource, config.window1, config.window2, discretizer, attributeIndices, driftType)
    )
  }

  def getMapData(dataSource: DataSource, range1: WindowRange, range2: WindowRange, discretizer: DiscretizeDateNum,
                 attributeIndices: Array[Int], driftType: String): Array[Array[Double]] = {
    val instances1 = getInstances(dataSource, discretizer, range1)
    val instances2 = getInstances(dataSource, discretizer, range2)
    val resultMap1 = StaticData.getResults(instances1, instances2, 1, attributeIndices, 1, 1, DriftMeasurement.valueOf(driftType.toUpperCase), 1)
    val resultMap2 = StaticData.getResults(instances1, instances2, 2, attributeIndices, 1, 1, DriftMeasurement.valueOf(driftType.toUpperCase), 1)
    resultMapToMapData(resultMap1, resultMap2, attributeIndices)
  }

  private def getInstances(dataSource: DataSource, discretizer: DiscretizeDateNum, bounds: WindowRange): Instances = {
    dataSource.hasMoreElements(dataSource.getStructure)
    dataSource.reset()
    var instances: Instances = new Instances(discretizer.getDiscreteStructure, bounds.start - bounds.end)
    val range = Range(bounds.start, bounds.end)
    iterateData(dataSource, instances, range, 0, discretizer)
  }

  private def iterateData(dataSource: DataSource, instances: Instances, range: Range, currentIndex: Int,
                          discretizeDateNum: DiscretizeDateNum): Instances = {
    val inst: Instance = dataSource.nextElement(dataSource.getStructure)
    inst match {
      case null => instances
      case _ =>
        if (range.contains(currentIndex)) instances.add(discretizeDateNum.discretizeInstance(inst))
        iterateData(dataSource, instances, range, currentIndex+1, discretizeDateNum)
    }
  }

  private def resultMapToMapData(resultMap1: java.util.Map[Array[Int], ExperimentResult],
                         resultMap2: java.util.Map[Array[Int], ExperimentResult],
                         attributeIndices: Array[Int]): Array[Array[Double]] = {
    // Define matrix
    var mapData: Array[Array[Double]] =
      (for (_ <- attributeIndices.indices)
        yield new Array[Double](attributeIndices.length))
        .toArray
    val keySet1 = resultMap1.keySet().toArray(new Array[Array[Int]](0))
    val keySet2 = resultMap2.keySet().toArray(new Array[Array[Int]](0))
    // Results for 2 attributes
    println(attributeIndices.deep.mkString(","))
    println(keySet2.deep.mkString(","))
    for (key:Array[Int] <- keySet2)
      yield
        mapData(attributeIndices.indexOf(key(0)))(attributeIndices.indexOf(key(1))) = resultMap2.get(key).getDistance
    // Symmetry
    for (key:Array[Int] <- keySet2)
      yield
        mapData(attributeIndices.indexOf(key(1)))(attributeIndices.indexOf(key(0))) = resultMap2.get(key).getDistance
    // Diagonal
    for (key:Array[Int] <- keySet1)
      yield
        mapData(attributeIndices.indexOf(key(0)))(attributeIndices.indexOf(key(0))) = resultMap1.get(key).getDistance
    mapData
  }
}

