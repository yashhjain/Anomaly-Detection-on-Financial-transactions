package com.spark.anomaly

import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.rdd.RDD

/**
  * Anomaly Detection algorithm
  */
class AnomalyDetection extends Serializable {

  val default_epsilon: Double = 0.1

  def run(data: RDD[Vector]): AnomalyDetectionModel = {
    val sc = data.sparkContext

    val statistic: MultivariateStatisticalSummary = Statistics.colStats(data)
    val mean: Vector = statistic.mean
    val variance: Vector = statistic.variance


    new AnomalyDetectionModel(mean, variance, default_epsilon)
  }

  /**
    * Uses the labeled input points to optimize the epsilon parameter by finding the best F1 Score
    * @param crossVal_data
    * @param an_Model
    * @return
    */
  def optimize(crossVal_data: RDD[LabeledPoint], an_Model: AnomalyDetectionModel) = {
    val sc = crossVal_data.sparkContext
    val bc_Mean = sc.broadcast(an_Model.means)
    val bc_Var = sc.broadcast(an_Model.variances)

    //compute probability density function for each example in the cross validation set
    val prob_CV: RDD[Double] = crossVal_data.map(labeledpoint =>
      AnomalyDetection.probFunction(labeledpoint.features, bc_Mean.value, bc_Var.value)
    )

    //select epsilon
    crossVal_data.persist()
    val epsilon_F1Score: (Double, Double) = evaluate(crossVal_data, prob_CV)
    crossVal_data.unpersist()

    //logInfo("Best epsilon %s F1 score %s".format(epsilonWithF1Score._1, epsilonWithF1Score._2))
    new AnomalyDetectionModel(an_Model.means, an_Model.variances, epsilon_F1Score._1)
  }

  /**
    *  Finds the best threshold to use for selecting outliers based on the results from a validation set and the ground truth.
    *
    * @param crossVal_data labeled data
    * @param prob_CV probability density function as calculated for the labeled data
    * @return Epsilon and the F1 score
    */
  private def evaluate(crossVal_data: RDD[LabeledPoint], prob_CV: RDD[Double]) = {

    val minP_val: Double = prob_CV.min()
    val maxP_val: Double = prob_CV.max()

    val sc = prob_CV.sparkContext

    var best_Epsilon = 0D
    var best_F1Score = 0D

    val step_size = (maxP_val - minP_val) / 1000.0


    for (ep <- minP_val to maxP_val by step_size){

      val bc_epsilon = sc.broadcast(ep)

      val cal_Predictions: RDD[Double] = prob_CV.map{ probs =>
        if (probs < bc_epsilon.value)
          1.0
        else
          0.0
      }
      val label_Prediction: RDD[(Double, Double)] = crossVal_data.map(_.label).zip(cal_Predictions)
      val label_Prediction_Cached: RDD[(Double, Double)] = label_Prediction

      val false_Positive = countStatisticalMeasure(label_Prediction_Cached, 0.0, 1.0)
      val true_Positive = countStatisticalMeasure(label_Prediction_Cached, 1.0, 1.0)
      val false_Negative = countStatisticalMeasure(label_Prediction_Cached, 1.0, 0.0)

      val precision = true_Positive / Math.max(1.0, true_Positive + false_Positive)
      val recall = true_Positive / Math.max(1.0, true_Positive + false_Negative)

      val f1_Score = 2.0 * precision * recall / (precision + recall)

      if (f1_Score > best_F1Score){
        best_F1Score = f1_Score
        best_Epsilon = ep
      }
    }

    (best_Epsilon, best_F1Score)
  }

  /**
    * Function to calculate true / false positives, negatives
    *
    * @param label_Prediction_Cached
    * @param lbl_Val
    * @param prediction_Val
    * @return
    */
  private def countStatisticalMeasure(label_Prediction_Cached: RDD[(Double, Double)], lbl_Val: Double, prediction_Val: Double): Double = {
    label_Prediction_Cached.filter { label_Prediction =>
      val lbl = label_Prediction._1
      val prediction = label_Prediction._2
      lbl == lbl_Val && prediction == prediction_Val
    }.count().toDouble
  }

}



object AnomalyDetection {


  /**
    * True if the given point is an anomaly, false otherwise
    * @param pt
    * @param mean
    * @param variance
    * @param ep
    * @return
    */
  private[anomaly] def predict (pt: Vector, mean: Vector, variance: Vector, ep: Double): Boolean = {
    probFunction(pt, mean, variance) < ep
  }

  private[anomaly] def probFunction(pt: Vector, mean: Vector, variance: Vector): Double = {
    val triplet_Feature: List[(Double, Double, Double)] = (pt.toArray, mean.toArray, variance.toArray).zipped.toList
    triplet_Feature.map { v =>
      val x = v._1
      val mean = v._2
      val variance = v._3
      val expectedValue = Math.pow(Math.E, -0.5 * Math.pow(x - mean,2) / variance)
      (1.0 / (Math.sqrt(variance) * Math.sqrt(2.0 * Math.PI))) * expectedValue
    }.product
  }





}



