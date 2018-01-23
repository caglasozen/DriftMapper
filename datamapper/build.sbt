name := """driftmap"""

version := "0.22-SNAPSHOT"

lazy val root = (project in file("."))
  .enablePlugins(PlayScala, RpmPlugin)

resolvers += Resolver.sonatypeRepo("snapshots")

scalaVersion := "2.12.2"

libraryDependencies += guice
libraryDependencies += filters
libraryDependencies += "org.scalatestplus.play" %% "scalatestplus-play" % "3.0.0" % Test
libraryDependencies += "com.h2database" % "h2" % "1.4.194"

libraryDependencies ++= Seq(
  "org.webjars" % "bootstrap" % "4.0.0",
  "org.webjars.npm" % "popper.js" % "1.13.0",
  "org.webjars.bower" % "plotly.js" % "1.27.1",
  "org.webjars.bower" % "resumable.js" % "2.11.2"
)

maintainer in Linux := "Loong Kuan Lee <lklee9@student.monash.edu>"
packageSummary in Linux := "Package for driftmap application in Linux"
packageDescription := "Allow other users to measure and analyse drift within a data set via a web application"
rpmRelease := "0.22"
rpmVendor := "driftmap.infotech.monash.edu.au"
rpmUrl := Some("https://github.com/LeeLoongKuan/DriftMapper.git")
rpmLicense := Some("Apache v2")