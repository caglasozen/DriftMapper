package services

import akka.actor.ActorRef
import global.DriftMeasurement
import play.api.libs.json.{JsValue, Json}
import play.api.mvc.Action

/**
  * Created by LoongKuan on 4/07/2017.
  */

class MessageAccumulator (out: ActorRef, interval: Int, driftMeasurement: DriftMeasurement) {
  var pointBacklog: Array[Int] = Array()
  var driftBacklog: Array[Array[Double]] = Array()
  var lastForward: Long = System.currentTimeMillis()

  def getAttachedActor: ActorRef = {
    this.out
  }

  def getDriftType: DriftMeasurement = {
    this.driftMeasurement
  }

  def addBacklog(point: Int, drifts: Array[Double], driftType: String): Unit = {
    // Add to backlog
    this.pointBacklog = this.pointBacklog :+ point
    if (this.driftBacklog.length == 0) {
      this.driftBacklog = for (drift <- drifts) yield Array(drift)
    }
    else {
      this.driftBacklog = (for (n <- drifts.indices) yield this.driftBacklog(n) :+ drifts(n)).toArray
    }

    // Check if sufficient time passed to forward backlog
    val timePassed = System.currentTimeMillis() - this.lastForward
    if (timePassed > this.interval) {
      forwardAll()
      this.lastForward = System.currentTimeMillis()
    }
  }

  def forwardAll(): Unit= {
    println("Sending Timeline Data")
    if (this.pointBacklog.length > 0) {
      this.out ! Json.obj(
        "messageType" -> "timelineUpdate",
        "value" -> Json.obj(
          "driftType" -> this.driftMeasurement.toString,
          "driftPoint" -> this.pointBacklog,
          "driftMagnitudes" -> this.driftBacklog)
      )
      this.pointBacklog = Array()
      this.driftBacklog = Array()
    }
  }
}

