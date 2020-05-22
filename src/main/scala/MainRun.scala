/* SimpleApp.scala */
import java.io.{File, PrintWriter}

import com.spark.anomaly.{AnomalyDetection, Parser}
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object MainRun {


  var op = ""

  def main(args: Array[String]) {

    if (args.length < 3) {
      println("Correct Usage: ProgramName Training_File CrossValidation_File Output_File")
      System.exit(1)
    }

    val train_fp = args(0)
    val crossVal_fp = args(1)
    val file_Object = new File(args(2))
    val print_Writer = new PrintWriter(file_Object)

    val config = new SparkConf().setAppName("Anomaly Detection Spark")
    val sc = new SparkContext(config)

    val train_data = sc.textFile(train_fp, 2).cache()
    val crossVal_data = sc.textFile(crossVal_fp, 2).cache()

    
    val train_rdd: RDD[Vector] = Parser.featureParse(train_data)
    val crossVal_rdd: RDD[LabeledPoint] = Parser.featureParseWithLabel(crossVal_data)

    val train_cached = train_rdd.cache()
    val an: AnomalyDetection = new AnomalyDetection()

    val model = an.run(train_cached)

    val cv_cached = crossVal_rdd.cache()
    val optimized_Model = an.optimize(cv_cached, model)


    val crossVal_vector = crossVal_rdd.map(_.features)
    val output = optimized_Model.predict(crossVal_vector)
    val outliers = output.filter(_._2).collect()
    //outliers.foreach(v => println(v._1))
    //println("\nFound %s outliers\n".format(outliers.length))
    outliers.foreach(v => op = op + v._1.toString + "\n")
    op = op + "\nFound %s outliers\n".format(outliers.length)

    print_Writer.write(op)
    print_Writer.close()
  }

}