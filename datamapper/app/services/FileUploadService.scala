package services

import java.io.{ File, RandomAccessFile }
import model._

import java.util.concurrent.{ ConcurrentMap, ConcurrentHashMap }
import scala.collection.JavaConverters._

class FileUploadService(serviceSavePath: String) {

  val basePath = if (serviceSavePath.endsWith("/")) {
    serviceSavePath
  } else { serviceSavePath + "/" }

  val uploadedParts: ConcurrentMap[String, Set[FileUploadInfo]] = new ConcurrentHashMap(8, 0.9f, 1)

  def fileNameFor(fileInfo: FileUploadInfo) = {
    s"${basePath}${fileInfo.resumableIdentifier}-${fileInfo.resumableFilename}"
  }

  def isLast(fileInfo: FileUploadInfo): Boolean = {
    (fileInfo.resumableTotalSize - (fileInfo.resumableChunkSize * fileInfo.resumableChunkNumber)) < fileInfo.resumableChunkSize
  }

  def savePartialFile(filePart: Array[Byte], fileInfo: FileUploadInfo) {
    if (filePart.length != fileInfo.resumableChunkSize & !isLast(fileInfo)) {
      println("error uploading part")
      return
    }
	val file = new File(fileNameFor(fileInfo))
	var bool = file.getParentFile().mkdirs();
	file.createNewFile()
    val partialFile = new RandomAccessFile(fileNameFor(fileInfo), "rw")
    val offset = (fileInfo.resumableChunkNumber - 1) * fileInfo.resumableChunkSize

    try {
      partialFile.seek(offset)
      partialFile.write(filePart, 0, filePart.length)
    } finally {
      partialFile.close()
    }

    val key = fileNameFor(fileInfo)
    if (uploadedParts.containsKey(key)) {
      val partsUploaded = uploadedParts.get(key)
      uploadedParts.put(key, partsUploaded + fileInfo)
    } else {
      uploadedParts.put(key, Set(fileInfo))
    }
  }

  def isPartialUploadComplete(fileInfo: FileUploadInfo): Boolean = {
    /* Todo: Should we open the RAF and check it? */
    val key = fileNameFor(fileInfo)
    uploadedParts.containsKey(key) && uploadedParts.get(key).contains(fileInfo)
  }

  def isUploadComplete(fileInfo: FileUploadInfo): Boolean = {
    val key = fileNameFor(fileInfo)
    val possibleFinishedFile = new RandomAccessFile(key, "r")
    val fileLength = possibleFinishedFile.length()
    fileLength == fileInfo.resumableTotalSize && uploadedParts.containsKey(key) && uploadedParts.get(key).size == fileInfo.totalChunks.toInt
  }

}