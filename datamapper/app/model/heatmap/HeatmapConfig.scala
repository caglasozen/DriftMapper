/**
  * Created by LoongKuan on 6/08/2017.
  */

package model.heatmap

case class HeatmapConfig(driftTypes: Array[String],
                         attributes: Array[String],
                         classAttribute: String,
                         window1: WindowRange,
                         window2: WindowRange
                         )
