package com.spark.anomaly

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD



object Parser{
  def featureParse(train_data: RDD[String]): RDD[Vector] = {
    val train_rdd: RDD[Array[Double]] = train_data.map(_.split(",").map(_.toDouble))
    val train_vector: RDD[Vector] = train_rdd.map(arrDouble => Vectors.dense(arrDouble))
    train_vector
  }

  def featureParseWithLabel(crossVal_data: RDD[String]): RDD[LabeledPoint] = {
    val cv_rdd: RDD[Array[Double]] = crossVal_data.map(_.split(",").map(_.toDouble))
    val cv_vector = cv_rdd.map(arrDouble => new LabeledPoint(arrDouble(0), Vectors.dense(arrDouble.slice(1, arrDouble.length))))
    cv_vector
  }
}
