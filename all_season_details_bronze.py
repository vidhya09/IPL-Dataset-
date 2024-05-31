# Databricks notebook source
spark

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

all_season_details_df = spark.read.option("header","true").option("inferSchema","true").csv("dbfs:/FileStore/shared_uploads/sivavithyaashree.m@thoughtworks.com/IPL_DATA/all_season_details.csv")
display(all_season_details_df)

# COMMAND ----------

all_season_details_df.printSchema()

# COMMAND ----------

changed_df = all_season_details_df\
    .withColumn("comment_id",col("comment_id").cast("int"))\
        .withColumn("season",col("season").cast("int"))\
            .withColumn("match_id",col("match_id").cast("long"))\
                    .withColumn("innings_id",col("innings_id").cast("int"))\
                        .withColumn("over",col("over").cast("int"))\
                            .withColumn("ball",col("ball").cast("int"))\
                                .withColumn("runs",col("runs").cast("int"))\
                                    .withColumn("isBoundary",col("isBoundary").cast("boolean"))\
                                        .withColumn("isWide",col("isWide").cast("boolean"))\
                                            .withColumn("isNoball",col("isNoball").cast("boolean"))\
                                                .withColumn("batsman1_id",col("batsman1_id").cast("long"))\
                                                    .withColumn("batsman1_runs",col("batsman1_runs").cast("int"))\
                                                        .withColumn("batsman1_balls",col("batsman1_balls").cast("int"))\
                                                            .withColumn("bowler1_id",col("bowler1_id").cast("long"))\
                                                                .withColumn("bowler1_overs",col("bowler1_overs").cast("float"))\
                                                                    .withColumn("bowler1_maidens",col("bowler1_maidens").cast("int"))\
                                                                        .withColumn("bowler1_runs",col("bowler1_runs").cast("int"))\
                                                                            .withColumn("bowler1_wkts",col("bowler1_wkts").cast("int"))\
                                                                                .withColumn("batsman2_id",col("batsman2_id").cast("long"))\
                                                                                    .withColumn("batsman2_runs",col("batsman2_runs").cast("int"))\
                                                        .withColumn("batsman2_balls",col("batsman2_balls").cast("int"))\
                                                            .withColumn("bowler2_id",col("bowler2_id").cast("long"))\
                                                                .withColumn("bowler2_overs",col("bowler2_overs").cast("float"))\
                                                                    .withColumn("bowler2_maidens",col("bowler2_maidens").cast("int"))\
                                                                        .withColumn("bowler2_runs",col("bowler2_runs").cast("int"))\
                                                                            .withColumn("bowler2_wkts",col("bowler2_wkts").cast("int"))\
                                                                                .withColumn("wicket_id",col("wicket_id").cast("double"))\
                                                                                    .withColumn("wkt_batsman_runs",col("wkt_batsman_runs").cast("float"))\
                                                                                        .withColumn("wkt_batsman_balls",col("wkt_batsman_balls").cast("float"))\
                                                                                        .withColumn("isRetiredHurt",col("isRetiredHurt").cast("boolean"))\



                                                        
changed_df.printSchema()
display(changed_df)

# COMMAND ----------

changed_df.count()

# COMMAND ----------

changed_df_no_null = changed_df.where(col("season").isNotNull())
display(changed_df_no_null)

# COMMAND ----------

# DBTITLE 1,Best Batsman in each season
batsman_runs = changed_df_no_null.where(col("batsman1_name").contains("Rahul Tripathi")).where(col("season").contains(2023)).groupBy("season","match_name","batsman1_name").agg(count("batsman1_runs").alias("batsman1_runs_count")).orderBy(col("season").desc(),col("match_name").desc())
display(batsman_runs)

# COMMAND ----------

display(changed_df_no_null.groupBy("season").agg(countDistinct("batsman1_name").alias("batsman1_name_count"),countDistinct("batsman2_name").alias("batsman2_name_count")).orderBy(col("season").asc()))

# COMMAND ----------

display(changed_df_no_null.groupBy("season","batsman1_name").agg(max("batsman1_name")).orderBy(col("season").desc()).where(col("season").contains(2023)))
display(changed_df_no_null.groupBy("season","batsman2_name").agg(max("batsman2_name")).orderBy(col("season").desc()).where(col("season").contains(2023)))

# COMMAND ----------

display(changed_df_no_null.groupBy("season","batsman1_name","batsman1_runs").agg(max("batsman1_runs")).orderBy(col("season").desc()).where(col("season").contains(2023)))
display(changed_df_no_null.groupBy("season","batsman2_name","batsman2_runs").agg(max("batsman2_runs")).orderBy(col("season").desc()).where(col("season").contains(2023)))

# COMMAND ----------

display(changed_df_no_null.groupBy("season","batsman1_name","batsman1_runs","batsman2_name","batsman2_runs").agg(max("batsman1_runs"),max("batsman2_runs")).orderBy(col("season").desc()).where(col("season").contains(2023)))