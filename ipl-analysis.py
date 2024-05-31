# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# COMMAND ----------

dff = pd.read_csv('/kaggle/input/indian-premier-league-ipl-all-seasons/all_season_summary.csv')
df = dff.copy()
df.head()

# COMMAND ----------

# Get basic information about the dataframe
print(df.info())

# COMMAND ----------

# Find the number of missing values in each column
df.isnull().sum()

# COMMAND ----------

# '1st_inning_score' is a crucial piece of information for our analysis
# Given the small number of missing values, we choose to drop these rows
df = df.dropna(subset=['1st_inning_score'])

# COMMAND ----------

# Display the data types of all columns
df.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC From the output, we can see that the '1st_inning_score' column is of type 'object'

# COMMAND ----------

# Split '1st_inning_score' into 'runs_scored_1st_inning' and 'wickets_lost_1st_inning'
df[['runs_scored_1st_inning', 'wickets_lost_1st_inning']] = df['1st_inning_score'].str.split('/', expand=True)

# Convert 'runs_scored_1st_inning' and 'wickets_lost_1st_inning' to numeric
df['runs_scored_1st_inning'] = pd.to_numeric(df['runs_scored_1st_inning'])
df['wickets_lost_1st_inning'] = pd.to_numeric(df['wickets_lost_1st_inning'])

# Delete the '1st_inning_score' column
df = df.drop(columns=['1st_inning_score'])

# Check the data types again to confirm
df.dtypes

# COMMAND ----------

# Split '2nd_inning_score' into 'runs_scored_2nd_inning' and 'wickets_lost_2nd_inning'
df[['runs_scored_2nd_inning', 'wickets_lost_2nd_inning']] = df['2nd_inning_score'].str.split('/', expand=True)

# Convert 'runs_scored_2nd_inning' and 'wickets_lost_2nd_inning' to numeric
df['runs_scored_2nd_inning'] = pd.to_numeric(df['runs_scored_2nd_inning'])
df['wickets_lost_2nd_inning'] = pd.to_numeric(df['wickets_lost_2nd_inning'])

# Delete the '2nd_inning_score' column
df = df.drop(columns=['2nd_inning_score'])

# COMMAND ----------

# Convert 'start_date' and 'end_date' to datetime
df['start_date'] = pd.to_datetime(df['start_date'])
df['end_date'] = pd.to_datetime(df['end_date'])

# Convert 'points' to numeric
df['points'] = pd.to_numeric(df['points'], errors='coerce')

# COMMAND ----------

# Rename columns for better understanding
df = df.rename(columns={
    'name': 'match_name',
    'short_name': 'match_short_name',
    'decision': 'toss_decision',
    'home_overs': 'home_team_overs',
    'home_runs': 'home_team_runs',
    'home_wickets': 'home_team_wickets_lost',
    'home_boundaries': 'home_team_boundaries',
    'away_overs': 'away_team_overs',
    'away_runs': 'away_team_runs',
    'away_wickets': 'away_team_wickets_lost',
    'away_boundaries': 'away_team_boundaries',
    'pom': 'player_of_match'
})

# COMMAND ----------

# Create new features
df['total_score'] = df['home_team_runs'] + df['away_team_runs']
df['score_difference'] = df['home_team_runs'] - df['away_team_runs']
df['total_wickets'] = df['home_team_wickets_lost'] + df['away_team_wickets_lost']
df['wickets_difference'] = df['home_team_wickets_lost'] - df['away_team_wickets_lost']
df['match_duration'] = (df['end_date'] - df['start_date']).dt.total_seconds() / 60
df['toss_advantage'] = (df['toss_won'] == df['winner']).astype(int)
df['home_advantage'] = (df['home_team'] == df['winner']).astype(int)
df['total_boundaries'] = df['home_team_boundaries'] + df['away_team_boundaries']
df['boundaries_difference'] = df['home_team_boundaries'] - df['away_team_boundaries']
df['run_rate'] = df['total_score'] / (df['home_team_overs'] + df['away_team_overs'])
df['wicket_rate'] = df['total_wickets'] / (df['home_team_overs'] + df['away_team_overs'])
df['match_day'] = df['start_date'].dt.dayofweek
df['match_month'] = df['start_date'].dt.month
df['home_team_performance'] = df.groupby('home_team')['home_team_runs'].transform('mean')
df['away_team_performance'] = df.groupby('away_team')['away_team_runs'].transform('mean')

# Check the dataframe to confirm
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC - Created 'total_score' as the sum of the 'home_team_runs' and 'away_team_runs'.
# MAGIC - Created 'score_difference' as the difference between the 'home_team_runs' and 'away_team_runs'.
# MAGIC - Created 'total_wickets' as the sum of the 'home_team_wickets_lost' and 'away_team_wickets_lost'.
# MAGIC - Created 'wickets_difference' as the difference between the 'home_team_wickets_lost' and 'away_team_wickets_lost'.
# MAGIC - Created 'match_duration' as the duration of each match.
# MAGIC - Created 'toss_advantage' to indicate whether the team that won the toss also won the match.
# MAGIC - Created 'home_advantage' to indicate whether the home team won the match.
# MAGIC - Created 'total_boundaries' as the sum of the 'home_team_boundaries' and 'away_team_boundaries'.
# MAGIC - Created 'boundaries_difference' as the difference between the 'home_team_boundaries' and 'away_team_boundaries'.
# MAGIC - Created 'run_rate' as the total runs scored divided by the total overs bowled.
# MAGIC - Created 'wicket_rate' as the total wickets taken divided by the total overs bowled.
# MAGIC - Created 'match_day' to indicate the day of the week when each match was played.
# MAGIC - Created 'match_month' to indicate the month of the year when each match was played.
# MAGIC - Created 'home_team_performance' as the average runs scored by each home team in their previous matches.
# MAGIC - Created 'away_team_performance' as the average runs scored by each away team in their previous matches.
# MAGIC
# MAGIC These new features will provide additional insights into the matches.

# COMMAND ----------

# Summary Statistics
summary_stats = df.describe(include='all').transpose()
summary_stats

# COMMAND ----------

# MAGIC %md
# MAGIC - The dataset contains 950 matches, spanning from the 2008 season to the 2022 season.
# MAGIC - The most frequent match is 'Chennai Super Kings v Mumbai Indians', which occurred 18 times.
# MAGIC - The team that won the toss most frequently is 'MI' (Mumbai Indians), which won the toss 123 times.
# MAGIC - The most common toss decision is 'BOWL FIRST', which occurred 599 times.
# MAGIC - The most common home score is '187/5', which occurred 10 times.

# COMMAND ----------

# Count Values
value_counts = df.nunique().to_frame('unique_values').sort_values(by='unique_values', ascending=False)
value_counts

# COMMAND ----------

# MAGIC %md
# MAGIC - The 'description' and 'id' columns have unique values for each match, as expected.
# MAGIC - The 'result' column has a relatively high number of unique values, indicating a variety of match outcomes.

# COMMAND ----------

# Missing Values
missing_values = df.isnull().sum().to_frame('missing_values').sort_values(by='missing_values', ascending=False)
missing_values

# COMMAND ----------

# MAGIC %md
# MAGIC - The 'wickets_lost_2nd_inning' and 'wickets_lost_1st_inning' columns have 105 and 65 missing values respectively. These missing values could represent matches where no wickets were lost in the respective innings.
# MAGIC - The 'highlights' column has 22 missing values. This could represent matches for which highlights are not available.
# MAGIC - The 'home_key_bowler' and 'away_key_bowler' columns have 13 and 11 missing values respectively. This could represent matches where a key bowler was not identified or not available.

# COMMAND ----------

# Outliers
import seaborn as sns
import matplotlib.pyplot as plt

# Select specific columns
cols_to_check = ['total_score', 'score_difference', 'total_boundaries', 'boundaries_difference']

# Create boxplots for each selected column
for col in cols_to_check:
    plt.figure(figsize=(5, 5))
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - The 'total_score' column has some values that are significantly higher than the rest, indicating potential outliers. These could represent matches where a high number of runs were scored.
# MAGIC - The 'score_difference' column has some  outliers. These could represent matches where the score difference was unusually low.
# MAGIC - The 'total_boundaries' column has some  outliers. These could represent matches where a high number of boundaries were scored.
# MAGIC - The 'boundaries_difference' column does not appear to have any significant outliers.

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploratory Data Analysis (EDA)

# COMMAND ----------

# Frequency Distribution of Categorical Variables

# Select categorical columns
cols_bar_plot = ['home_team', 'away_team', 'toss_won', 'toss_decision', 'winner', 'venue_name', 'home_captain', 'away_captain','umpire1', 'umpire2', 'tv_umpire', 'referee', 'reserve_umpire']

# Plot frequency distribution for each categorical column
for col in cols_bar_plot:
    plt.figure(figsize=(10, 5))
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Frequency Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# COMMAND ----------

# Frequency Distribution of Selected Categorical Variables

# Select specific categorical columns
selected_categorical_cols = ['toss_decision', 'winner']

# Plot frequency distribution for each selected categorical column
for col in selected_categorical_cols:
    plt.figure(figsize=(10, 5))
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Frequency Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - 'toss_decision': The 'BOWL FIRST' decision is more frequent than the 'BAT FIRST' decision. This could suggest that teams prefer to bowl first after winning the toss.
# MAGIC - 'winner': The team that wins most frequently is 'MI' (Mumbai Indians), followed by 'CSK' (Chennai Super Kings) and 'KKR' (Kolkata Knight Riders). This could suggest that these teams have been more successful in the matches.

# COMMAND ----------

# Histograms of Numerical Variables

# Select specific numerical columns
selected_numerical_cols = ['total_score', 'score_difference', 'total_boundaries', 'boundaries_difference']

# Plot histogram for each selected numerical column
for col in selected_numerical_cols:
    plt.figure(figsize=(10, 5))
    df[col].hist(bins=30)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - 'total_score': The distribution is slightly right-skewed, indicating that there are some matches with exceptionally high total scores. The most common total score seems to be around 300-350. (total_score is sum of runs scored by both teams in that match)
# MAGIC - 'score_difference': The distribution is approximately symmetric, indicating that the score difference is equally likely to be positive or negative.
# MAGIC - 'total_boundaries': The distribution is slightly right-skewed, indicating that there are some matches with exceptionally high total boundaries. The most common total boundaries seems to be around 30-40. (total_boundaries is sum of boundaries scored by both teams in that match)
# MAGIC - 'boundaries_difference': The distribution is approximately symmetric, indicating that the boundaries difference is equally likely to be positive or negative. 

# COMMAND ----------

# Box Plots of Numerical Variables

# Plot box plot for each selected numerical column
for col in selected_numerical_cols:
    plt.figure(figsize=(10, 5))
    df[col].plot(kind='box')
    plt.title(f'Box Plot of {col}')
    plt.ylabel('Value')
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - 'total_score': The box plot shows that the median total score is around 300, and the interquartile range (IQR) is approximately 250-350. There are some outliers on the higher end, indicating some matches with exceptionally high total scores.
# MAGIC - 'score_difference': The box plot shows that the median score difference is around 0, and the IQR is approximately -50 to 50. There are some outliers on both ends, indicating some matches with exceptionally high positive or negative score differences.
# MAGIC - 'total_boundaries': The box plot shows that the median total boundaries is around 30, and the IQR is approximately 20-40. There are some outliers on the higher end, indicating some matches with exceptionally high total boundaries.
# MAGIC - 'boundaries_difference': The box plot shows that the median boundaries difference is around 0, and the IQR is approximately -10 to 10. There are some outliers on both ends, indicating some matches with exceptionally high positive or negative boundaries differences.

# COMMAND ----------

# Scatter Plot of 'total_score' vs 'total_boundaries'

plt.figure(figsize=(10, 5))
plt.scatter(df['total_score'], df['total_boundaries'])
plt.title('Scatter Plot of Total Score vs Total Boundaries')
plt.xlabel('Total Score')
plt.ylabel('Total Boundaries')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The scatter plot above shows the relationship between 'total_score' and 'total_boundaries'. There seems to be a positive correlation between the two variables, indicating that matches with higher total scores also tend to have more boundaries.

# COMMAND ----------

# Box Plot of 'total_score' across different 'winner' categories

plt.figure(figsize=(10, 5))
sns.boxplot(x='winner', y='total_score', data=df)
plt.title('Box Plot of Total Score across Winner Categories')
plt.xlabel('Winner')
plt.ylabel('Total Score')
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - Some teams like 'MI' (Mumbai Indians) and 'CSK' (Chennai Super Kings) have a wider interquartile range (IQR), indicating that the total scores in their matches are more varied.
# MAGIC - Some teams have outliers on the higher end, indicating some matches with exceptionally high total scores.

# COMMAND ----------

# Bar Plot of Mean 'total_score' across different 'winner' categories

# Calculate mean total score for each winner category
mean_total_score_by_winner = df.groupby('winner')['total_score'].mean()

plt.figure(figsize=(10, 5))
mean_total_score_by_winner.plot(kind='bar')
plt.title('Bar Plot of Mean Total Score across Winner Categories')
plt.xlabel('Winner')
plt.ylabel('Mean Total Score')
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - The mean total score varies across different winner teams, indicating that the total score is influenced by the winning team.
# MAGIC - Some teams like 'MI' (Mumbai Indians) and 'CSK' (Chennai Super Kings) have a higher mean total score, indicating that their matches tend to have higher scores.
# MAGIC - Some teams like 'RR' (Rajasthan Royals) and 'DC' (Delhi Capitals) have a lower mean total score, indicating that their matches tend to have lower scores.

# COMMAND ----------

# Heatmap of Correlation between 'total_score' and 'total_boundaries'

# Calculate correlation between 'total_score' and 'total_boundaries'
correlation = df[['total_score', 'total_boundaries']].corr()

plt.figure(figsize=(10, 5))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlation between Total Score and Total Boundaries')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The heatmap above shows the correlation between 'total_score' and 'total_boundaries'. The correlation coefficient is 0.89, indicating a strong positive correlation between the two variables. This means that matches with higher total scores also tend to have more boundaries. 

# COMMAND ----------

# Cross Tabulation of 'winner' and 'toss_decision'

cross_tab = pd.crosstab(df['winner'], df['toss_decision'])
cross_tab

# COMMAND ----------

# MAGIC %md
# MAGIC - For most teams, the number of wins is higher when they decide to bowl first after winning the toss. This could indicate that bowling first might be a more successful strategy for these teams.
# MAGIC - However, for some teams like 'CSK' (Chennai Super Kings), the number of wins is almost equal regardless of the toss decision. This could indicate that the toss decision does not significantly affect the match outcome for these teams.

# COMMAND ----------

# Pair Plot of 'total_score', 'total_boundaries', 'score_difference', and 'boundaries_difference'

sns.pairplot(df[['total_score', 'total_boundaries', 'score_difference', 'boundaries_difference']])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - 'total_score' and 'total_boundaries' have a strong positive correlation, as seen in the scatter plot and also confirmed by the correlation coefficient calculated earlier. This indicates that matches with higher total scores also tend to have more boundaries.
# MAGIC - 'score_difference' and 'boundaries_difference' also seem to have a positive correlation, indicating that matches with higher score differences also tend to have higher boundaries differences.

# COMMAND ----------

# Correlation Matrix of All Numeric Variables

# Calculate correlation matrix
correlation = df.corr()

# Plot heatmap of correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlation Matrix')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - 'total_score' and 'score_difference' have a weak negative correlation (-0.07), indicating that matches with higher total scores do not necessarily have higher score differences.
# MAGIC - 'total_boundaries' and 'boundaries_difference' have a weak negative correlation (-0.06), indicating that matches with more boundaries do not necessarily have higher boundaries differences.

# COMMAND ----------

# Grouped Box Plots of 'total_score' across different 'winner' and 'toss_decision' categories

plt.figure(figsize=(15, 10))
sns.boxplot(x='winner', y='total_score', hue='toss_decision', data=df)
plt.title('Grouped Box Plots of Total Score across Winner and Toss Decision Categories')
plt.xlabel('Winner')
plt.ylabel('Total Score')
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - For most teams, the distribution of total score is similar regardless of the toss decision, indicating that the toss decision does not significantly affect the total score.
# MAGIC - However, for some teams like 'MI' (Mumbai Indians) and 'CSK' (Chennai Super Kings), the total score seems to be slightly higher when they decide to bowl first after winning the toss.
# MAGIC - The interquartile range (IQR) and the presence of outliers vary across different winner teams, indicating that the total score is influenced by the winning team.

# COMMAND ----------

# Pivot Table of 'total_score' across different 'winner' and 'toss_decision' categories

pivot_table = df.pivot_table('total_score', index='winner', columns='toss_decision', aggfunc='mean')
pivot_table

# COMMAND ----------

# MAGIC %md
# MAGIC - For most teams, the mean total score is similar regardless of the toss decision, indicating that the toss decision does not significantly affect the total score.
# MAGIC - However, for some teams like 'MI' (Mumbai Indians) and 'CSK' (Chennai Super Kings), the mean total score seems to be slightly higher when they decide to bowl first after winning the toss.

# COMMAND ----------

# Scatter Plot of 'total_score' and 'total_boundaries' with points colored by 'winner'

plt.figure(figsize=(10, 10))
sns.scatterplot(x='total_score', y='total_boundaries', hue='winner', data=df)
plt.title('Scatter Plot of Total Score and Total Boundaries colored by Winner')
plt.xlabel('Total Score')
plt.ylabel('Total Boundaries')
plt.show()