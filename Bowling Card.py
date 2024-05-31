# Databricks notebook source
from pyspark.sql.functions import *

# COMMAND ----------

#Import data
ipl_df = spark.read.format("csv") \
    .option("header","true") \
        .option("inferSchema","true") \
            .load("/FileStore/shared_uploads/sivavithyaashree.m@thoughtworks.com/IPL_DATA/all_season_bowling_card.csv")

display(ipl_df)

# COMMAND ----------

ipl_df.printSchema()

# COMMAND ----------

#Check for Null
ipl_df=ipl_df.where(col("season").isNotNull())
display(ipl_df)

# COMMAND ----------

from pyspark.sql.functions import col

# Selecting only the name and dots columns
bowler_ipl_df = ipl_df.select("name", "dots")

# Sort by dots in ascending order
bowler_ipl_df = bowler_ipl_df.orderBy(col("dots").desc())

# Display the dataframe
display(bowler_ipl_df)


# COMMAND ----------

from pyspark.sql.window import Window

# Define a window partitioned by season and ordered by total wickets in descending order
window_spec = Window.partitionBy("season")

# Calculate total wickets for each player in each season
bowler_ipl_df = ipl_df.groupBy("name", "season").agg(sum(col("wickets")).alias("total_wickets"))

# Use window function to get the player with the most wickets in each season
bowler_ipl_df = bowler_ipl_df.withColumn("max_wickets", max(col("total_wickets")).over(window_spec)) \
    .filter(col("total_wickets") == col("max_wickets")) \
    .drop("max_wickets")

# Display the dataframe
display(bowler_ipl_df.orderBy(col("season").desc()))

# COMMAND ----------

bowler_ipl_df = ipl_df.groupBy("season","name").agg(sum(col("wickets")).alias("total_wickets"))
bowler_ipl_df = bowler_ipl_df.orderBy(col("total_wickets").desc())
display(bowler_ipl_df)