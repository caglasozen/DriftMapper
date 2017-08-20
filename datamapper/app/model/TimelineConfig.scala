package model

case class TimelineConfig( classAttribute: String,
                           modelType: String,
                           increment: Boolean,
                           groupSize: Int,
                           driftTypes: Array[String],
                           subsetLength: Int,
                           attributes: Array[String],
                           groupAttribute: String
                         )