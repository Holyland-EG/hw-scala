package homework
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.log4j._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg._
import org.apache.spark.sql.expressions.Window
import java.io.{BufferedWriter, File, FileWriter}

object main extends App {
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)
  val spark: SparkSession = SparkSession.builder()
    .master("local[1]")
    .appName("Recommendations")
    .getOrCreate()

  val path = "src/main/DO_record_per_line.json"
  val courses = spark.read.json(path)

  val courses_clean = courses
    .withColumn("desc", regexp_replace(col("desc"), "[^\\w\\sА-Яа-яёй]+", " "))
    .withColumn("desc", regexp_replace(col("desc"), "[0-9_]+", " "))
    .withColumn("desc", regexp_replace(col("desc"), "\\s+", " "))
    .where(length(col("desc")) > 0)

  val tokenizer = new Tokenizer().setInputCol("desc").setOutputCol("words")
  val wordsData = tokenizer.transform(courses_clean)

  val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(10000)
  val featurizedData = hashingTF.transform(wordsData)

  val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
  val idfModel = idf.fit(featurizedData)
  val rescaledData = idfModel.transform(featurizedData)

  val Dense = udf((v: Vector) => v.toDense)
  val df = rescaledData.withColumn("dense_features", Dense(col("features")))

  val cosSimilarity = udf { (x: Vector, y: Vector) =>
    val v1 = x.toArray
    val v2 = y.toArray
    val l1 = scala.math.sqrt(v1.map(x => x*x).sum)
    val l2 = scala.math.sqrt(v2.map(x => x*x).sum)
    val scalar = v1.zip(v2).map(p => p._1*p._2).sum
    scalar/(l1*l2)
  }

  val var_11 = Seq(387, 15, 56, 811, 1443, 2011)

  val filtered_df = df
    .filter(col("id").isin(var_11: _*))
    .select(col("id").alias("id_1"), col("dense_features").alias("dense_1"), col("lang").alias("lang_1"))

  val joinedDf = df.join(broadcast(filtered_df), col("id") =!= col("id_1") && col("lang") === col("lang_1"))
    .withColumn("cosine_sim", cosSimilarity(col("dense_1"), col("dense_features")))

  val w = Window.partitionBy(col("id_1")).orderBy(col("cosine_sim").desc, col("name").asc, col("id").asc)

  val filtered = joinedDf
    .withColumn("cosine_sim", when(col("cosine_sim").isNaN, 0).otherwise(col("cosine_sim")))
    .withColumn("rank", row_number().over(w))
    .filter(col("rank")between(2, 11))

  val output_file = new File("/Users/evagolubenko/Downloads/recommendations/src/main/scala/output.json")
  val bw = new BufferedWriter(new FileWriter(output_file))
  bw.write("{\n")

  // filtered.select("id", "id_1").groupBy("id_1").agg(collect_list("id"))

  for ( i <- var_11 ) {
    val recs = filtered.select("id").where(col("id_1") === i).rdd.map(r => r(0)).collect()
    bw.write("\"" + i + "\":[")
    for ( value <- recs ) {
      if (value == recs.last) {
        bw.write("" + value + "")
      } else {
        bw.write("" + value + ", ")
      }
    }
    if( i == var_11.last ){
      bw.write("]\n")
    } else {
      bw.write("],\n")
    }
  }
  bw.write("}")
  bw.close()
}