

Term Project Final Report
2022/06/04
데이터과학(14455001) 
201732831 오현호
<Overall Source code>
import pandas as pd
import numpy as np

pd.set_option('display.max_row',20000)
# pd.set_option('display.max_column', 2000)
df = pd.read_csv('C:\Python_DataScience\data\soccer\players_21.csv', encoding='unicode_escape')
df.info(verbose=True, null_counts=True)

# Remove the country except 32 World cup country
df = df[df['nationality'].isin(
    ['Ecuador', 'Senegal', 'Netherlands', 'England', 'Iran', 'United States', 'Wales',
     'Argentina', 'Saudi Arabia', 'Mexico', 'Poland', 'France', 'Denmark', 'Tunisia', 'Peru', 'Spain', 'Germany', 'Japan', 'Costa Rica', 'Belgium', 'Canada',
     'Morocco', 'Croatia', 'Brazil', 'Serbia', 'Switzerland', 'Cameroon', 'Portugal', 'Ghana', 'Uruguay',
     'Korea Republic'])]
df

# Select column
df = df[['short_name', 'nationality', 'potential', 'player_positions', 'nation_position', 'ls', 'st', 'rs', 'lw', 'lf',
         'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb',
         'lcb', 'cb', 'rcb', 'rb', 'overall']]
df.isnull().values.any()
df[df['player_positions'].isnull()]

df.loc[df['nation_position'] != df['nation_position'], 'nation_position'] = df['player_positions']
df.loc[df['nation_position'] == 'SUB', 'nation_position'] = df['player_positions']

CF = ['ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw']
MF = ['lam', 'ram', 'cam', 'lm', 'lcm', 'cm', 'rcm', 'rm']
DF = ['lwb', 'rwb', 'cdm', 'rdm', 'lb', 'lcb', 'rcb', 'rb', 'ldm', 'cb']
df['nation_position'] = df['nation_position'].astype('str')
position_list = df['nation_position'].str.split(',')
df['position'] = position_list.str.get(0)
df['nation_position'] = df['position']
df['nation_position'] = df['nation_position'].str.lower()
df['overall_'] = ''
df


for i in range(8):
    df.loc[df['nation_position'] == CF[i], 'overall_'] = df[CF[i]]
for i in range(8):
    df.loc[df['nation_position'] == MF[i], 'overall_'] = df[MF[i]]
for i in range(10):
    df.loc[df['nation_position'] == DF[i], 'overall_'] = df[DF[i]]
df.loc[df['overall_'] == '', 'overall_'] = df['overall']

for i in range(8):
    df.loc[df['nation_position'] == CF[i], 'nation_position'] = "CF"
for i in range(8):
    df.loc[df['nation_position'] == MF[i], 'nation_position'] = "MF"
for i in range(10):
    df.loc[df['nation_position'] == DF[i], 'nation_position'] = "DF"
df.loc[df['nation_position'] == 'gk', 'nation_position'] = "GK"

df.sort_values(by=['nationality'],inplace =True)
df

df = df[['short_name', 'nationality', 'potential', 'nation_position', 'overall','overall_']]
country_list = ['Ecuador','Senegal','Netherlands','England','Iran','United States','Wales','Ukraine','Ukraine','Scotland','Argentina','Saudi Arabia','Mexico','Poland','France','Denmark','Tunisia','Australia','United Arab Emirates','Peru','Spain','Germany','Japan','Costa Rica','New Zealand','Belgium','Canada','Morocco','Croatia','Brazil','Serbia','Switzerland','Cameroon','Portugal','Ghana','Uruguay','Korea Republic']
    
# df.loc[df['nation_position'] == DF[i], 'overall_'] = df[DF[i]]
# subDF = dfx[dfx['Product']=='Apples']
# df.loc[df['nation_position']

df_CF = pd.DataFrame()
df_MF = pd.DataFrame()
df_DF = pd.DataFrame()
df_total= pd.DataFrame()

#Calculate each of CF, MF, DF’s  overall 
#나라별 total overall 계산
for x in country_list:
       df2 = df[df['nationality']==x]
       CF_overall_top = df2.loc[df2['nation_position']=='CF',:].sort_values('overall', ascending =False).head(3)
       MF_overall_top = df2.loc[df2['nation_position']=='MF',:].sort_values('overall', ascending =False).head(3)
       DF_overall_top = df2.loc[df2['nation_position']=='DF',:].sort_values('overall', ascending =False).head(5)
       total_overall_top = df2.sort_values('overall', ascending =False).head(11)
       df_CF = pd.concat([df_CF,CF_overall_top],axis=0)
       df_MF = pd.concat([df_MF,MF_overall_top],axis=0)
       df_DF = pd.concat([df_DF,DF_overall_top],axis=0)
       df_total = pd.concat([df_total,total_overall_top],axis=0)

df_total

df_result_player = pd.concat([df_CF,df_MF,df_DF],axis=0).sort_values(['nationality','nation_position'],ascending =[True,True])
df_result_player
df_result_player2 = df.groupby(['nationality','nation_position'])['overall'].mean().reset_index(name='Overall(mean)')
df_result_player2
df_result_player3 = df.groupby('nationality')['overall'].mean().reset_index(name='overall(mean)')
df_CF2 = df_CF.groupby('nationality')['overall'].mean().reset_index(name='CF_Overall(mean)')
df_MF2 = df_MF.groupby('nationality')['overall'].mean().reset_index(name='MF_Overall(mean)')
df_DF2 = df_DF.groupby('nationality')['overall'].mean().reset_index(name='DF_Overall(mean)')
df_total2 = df_total.groupby('nationality')['overall'].mean().reset_index(name='Total_Overall(mean)')

for x in df_CF2:
       df_total2['Total_Overall(mean)'] = (df_CF2['CF_Overall(mean)'] + df_MF2['MF_Overall(mean)'] + df_DF2['DF_Overall(mean)'])/3

df_total2

df_result = pd.concat([df_CF2,df_MF2.iloc[:,1],df_DF2.iloc[:,1],df_total2.iloc[:,1]],axis=1)
df_result

df_result2 = df_result[['nationality','Total_Overall(mean)']]
df_result2.to_csv('C:\Python_DataScience\data\soccer\players_21_result.csv')




<Source code>
from google.colab import drive drive.mount('/content/drive')
# using graphviz 
import pydot
import pygraphviz

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydotplus
import graphviz
from sklearn.tree import export_graphviz
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import Image
import random

# data preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder

# linear regression
from sklearn.linear_model import LinearRegression

# classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
# Multi target regression.
#This strategy consists of fitting one regressor per target.
#parameters
# estimator: estimator object ( An estimator object implementing fit and predict )
# n_job: sint or None, optional (default=None) (The number of jobs to run in parallel. fit, predict # and partial_fit (if supported by the passed estimator) will be parallelized for each target.)

# clustering
from sklearn.cluster import KMeans

# data split, validation, parameter tuning
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

# evaluationi
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.metrics import accuracy_score

# warning
import warnings 
warnings.filterwarnings(action='ignore')

#Data Curation

#List of 32 countries qualifying for Qatar World Cup 2022
country_list = ['Ecuador', 'Senegal', 'Netherlands', 'England', 'Iran', 'United States', 'Wales','Argentina', 
                'Saudi Arabia', 'Mexico', 'Poland', 'France', 'Denmark', 'Tunisia', 'Peru', 'Spain',
                'Germany', 'Japan', 'Costa Rica', 'Belgium', 'Canada','Morocco', 'Croatia', 'Brazil', 
                'Serbia', 'Switzerland', 'Cameroon', 'Portugal', 'Ghana', 'Uruguay','South Korea', 'Qatar']

# read the dataset of international football matches played between countries around the world from 1872 to 2022
# https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
results = pd.read_csv('/content/drive/MyDrive/2022-1/데이터과학/Term Project/Winner/results.csv', encoding='unicode_escape')
results['date'] = pd.to_datetime(results['date']) # change object to datetime
results.head() 

# read the dataset of FIFA's official rankings by year
# https://www.kaggle.com/datasets/tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now
rankings = pd.read_csv('/content/drive/MyDrive/2022-1/데이터과학/Term Project/Winner/rankings.csv', encoding='unicode_escape')
rankings['rank_date'] = pd.to_datetime(rankings['rank_date']) # change object to datetime

# check the year distribution of the ranking data
years = []
for date in rankings.rank_date:
    years.append(int(str(date)[0:4]))
plt.figure(figsize=(14, 6))
plt.hist(years, density=True, bins=12, edgecolor="k")
plt.title("Histogram of Years")
plt.ylabel("Frequency")
plt.xlabel("Year")
plt.show() 

# read the dataset containing various information about soccer players
# https://www.kaggle.com/datasets/stefanoleone992/fifa-21-complete-player-dataset
overall = pd.read_csv('/content/drive/MyDrive/2022-1/데이터과학/Term Project/Winner/overall.csv', encoding='unicode_escape')

#Checking the distribution of the overall
plt.hist(overall['overall'],rwidth=0.9)
plt.xlabel('overall')
plt.ylabel('Count')
plt.show()

sns.distplot(overall['overall'])
overall['overall'].describe() 

# read the dataset of countries and opponents in 2022 FIFA World Cup Qatar
# datasets were created by ourselves
world_cup = pd.read_csv('/content/drive/MyDrive/2022-1/데이터과학/Term Project/Winner/World Cup 2022 Dataset.csv')
world_cup.head() 

# Data Inspection - data exploration

# friendly matches are the most common
# all players would play their best even in friendly matches
rank_bound = 20
match_sort = results.tournament.value_counts()[:rank_bound].sort_values()
value = match_sort.values
label = match_sort.index

# show with a barplot
plt.figure(figsize=(14, 6))
plt.barh(y=label, width=value, edgecolor="k")
for i in range(rank_bound):
    plt.text(x=50, y=i - 0.1, s=value[i], color="w", fontsize=12) 

# more recent matches, the more data
years = []

for date in results.date:
    years.append(int(str(date)[0:4]))

# show with a histogram
plt.figure(figsize=(14, 6))
plt.hist(years, density=True, bins=12, edgecolor="k")
plt.title("Histogram of Years")
plt.ylabel("Frequency")
plt.xlabel("Year")
plt.show() 

# winnig rate was much better when you were on the home team than away

home_team_ws = results.copy()

# only 32 countries that advanced to the Qatar World Cup finals are extracted
home_team_ws = home_team_ws[home_team_ws['home_team'].isin(country_list)]
home_team_ws = home_team_ws[home_team_ws['away_team'].isin(country_list)]

# categorize by conditions as wins, draws, and losses
conditions = [home_team_ws["home_score"] == home_team_ws["away_score"], home_team_ws["home_score"] > home_team_ws["away_score"],
              home_team_ws["home_score"] < home_team_ws["away_score"]]
choices = ["Draw", 'Win', 'Lost']
home_team_ws["Win_Statues"] = np.select(conditions, choices)

# show with a displot
sns.displot(home_team_ws, x="Win_Statues")
plt.title("Home Team Winning Status") 
# sort the winning percentages of each team in ascending order
# function that returns winning percentage when playing on home teams from 32 countries playing in Qatar World Cup
def win_prob(team,result):
    
    win_statues = results.copy()
    
    # only 32 countries that advanced to the Qatar World Cup finals are extracted
    win_statues = win_statues[win_statues['home_team'].isin(country_list)]
    win_statues = win_statues[win_statues['away_team'].isin(country_list)]

    # categorize by conditions as wins, drawin_statues, and losses
    conditions = [win_statues["home_score"] == win_statues["away_score"], win_statues["home_score"] > win_statues["away_score"],
              win_statues["home_score"] < win_statues["away_score"]]
    choices = ["Draw", 'Win', 'Lost']
    win_statues["Win_Statues"] = np.select(conditions, choices)
    
    teams_win_statues = pd.crosstab(win_statues[team], win_statues["Win_Statues"], margins=True, margins_name="Total")
    teams_win_statues["team_win_probability"] = teams_win_statues[result] / (teams_win_statues["Total"])

    # select teams which plays at least 50 games
    teams_win_statues_50 = teams_win_statues.loc[teams_win_statues["Total"] > 50]
    teams_win_statues_50 = teams_win_statues_50.sort_values("team_win_probability", ascending=False)
    return teams_win_statues_50

teams_home_statues = win_prob("home_team","Win")
teams_home_statues.style.bar(color="orange", subset="team_win_probability") 

# winnig rate of each team when the team is away
teams_away_statues = win_prob("away_team","Lost")
teams_away_statues.rename(columns={'Lost': 'Win'}, index={'Win': 'Lost'}, inplace=True)
teams_away_statues.head(10) 

# Data Preprocessing - data restructuring

# Wrong data
# names of countries, which are marked differently from the main dataset, are unified.

# result data
results =  results.replace({'Germany DR': 'Germany', "Korea Republic" : "South Korea", 'China': 'China PR'})

# ranking data
rankings = rankings.replace({"IR Iran": "Iran", "Korea Republic" : "South Korea", "USA" : "United States"}) 

# overall data
overall = overall.replace({"Korea Republic" : "South Korea"})

# world_cup match data
world_cup = world_cup.replace({"USA": "United States", "Korea Republic" : "South Korea", "IR Iran": "Iran"})
world_cup = world_cup.set_index('Team') # set index to 'Team' column

# Data Preprocessing - data restructuring

# Ranking dataset
# create a column that gives a ranking weight score by country
# current_year_ranking_point(100%) + two_years_ago_ranking_point(30%) + threee_years_ago_ranking_point(20%)
rankings['weighted_points'] =  rankings['cur_year_avg_weighted'] + rankings['two_year_ago_weighted'] + rankings['three_year_ago_weighted']
rankings = rankings.drop(['cur_year_avg_weighted', 'two_year_ago_weighted', 'three_year_ago_weighted'], axis = 1) # drop columns

# create a new colum named "Win_Statues" to store the result(Win,Lost,Draw) of the home team
conditions = [results["home_score"] == results["away_score"], results["home_score"] > results["away_score"],
              results["home_score"] < results["away_score"]]
choices = ["Draw", 'Win', 'Lost']
results["won"] = np.select(conditions, choices)
results_match=results.copy()

# Ranking by country, by date
# if there is missing value, replace it with ffill
rankings = rankings.set_index(['rank_date']).groupby(['country_full'], group_keys=False).resample('D').first().fillna(method='ffill').reset_index()

# add the ranking information of the home team to the main data set
results = results.merge(rankings,left_on=['date', 'home_team'],right_on=['rank_date', 'country_full'])
# add the ranking information of the away team to the main data set
results = results.merge(rankings, left_on=['date', 'away_team'], right_on=['rank_date', 'country_full'], suffixes=('_home', '_away'))

# Overall dataset
# add the overall information of the home team to the main data set
results = results.merge(overall,left_on='home_team',right_on='nationality', how = 'outer')
# add the overall information of the away team to the main data set
results = results.merge(overall,left_on='away_team',right_on='nationality',how = 'outer', suffixes=('_home', '_away'))

# Results dataset
# create new columns to represent the difference between the two countries
results['rank_difference'] = results['rank_home'] - results['rank_away']
results['average_rank'] = (results['rank_home'] + results['rank_away'])/2
results['point_difference'] = results['weighted_points_home'] - results['weighted_points_away']
results['score_difference'] = results['home_score'] - results['away_score']
results['overall_difference'] = results['overall_home']- results['overall_away']
results['is_won'] = results['score_difference'] > 0
results.head() 

# since two data frames were outer joined, a lot of na was created
# fill na with zero
results['nationality_home'].fillna(0, inplace = True)
results['nationality_away'].fillna(0, inplace = True)
results['overall_home'].fillna(0, inplace = True)
results['overall_away'].fillna(0, inplace = True)
results['overall_difference'].fillna(0, inplace = True)

# Preprocessing - Data Value Changes - Cleaning dirty data - Missing data

print(results.isnull().sum()) 

# delete all rows with na
results = results.dropna(axis=0)

print(results.isnull().sum()) 

# make a copy of the dataframe for future use
main = results.copy()

#Preprocessing - Data Value Changes - Cleaning dirty data - unusable data

# only 32 countries that advanced to the Qatar World Cup finals are extracted
results = results[results['home_team'].isin(country_list)]
results = results[results['away_team'].isin(country_list)]

# delete all duplicate or unnecessary features
results.drop(['city'],axis=1,inplace=True)
results.drop(['tournament'],axis=1,inplace=True)
results.drop(['neutral'],axis=1,inplace=True)
results.drop(['rank_date_home'],axis=1,inplace=True)
results.drop(['country_full_home'],axis=1,inplace=True)
results.drop(['country_abrv_home'],axis=1,inplace=True)
results.drop(['rank_date_away'],axis=1,inplace=True)
results.drop(['country_full_away'],axis=1,inplace=True)
results.drop(['country_abrv_away'],axis=1,inplace=True)
results.drop(['nationality_home'],axis=1,inplace=True)
results.drop(['nationality_away'],axis=1,inplace=True)
results.info() 

#Preprocessing - Data Value Changes - Cleaning dirty data - outliers

# find outliers in the numerical data columns
fig = plt.figure(figsize=(10, 5))
plt.boxplot([results["home_score"], results["away_score"]])
plt.xticks([1, 2], ["Home Score", "Away Score"]) 

# Asume maximum goals that one team can score is 7
higher_home = 8
higher_away = 8
results = results[(results["home_score"] < higher_home) & (results["away_score"] < higher_away)]
score = results['home_score'] - results['away_score']

# show how the outlier has changed
fig = plt.figure(figsize=(10, 5))
plt.boxplot([results["home_score"], results["away_score"]])
plt.xticks([1, 2], ["Home Score", "Away Score"]) 

#Preprocessing - Data Value Changes - Data normalization

# StandardScaler
# standard scaling to match the range of numerical data similarly
X = results.copy()
scaler = StandardScaler()
X.loc[:,['overall_home','overall_away','home_score','away_score', 'weighted_points_home', 'rank_home', 'weighted_points_away', 
         'rank_away', 'rank_difference', 'average_rank','point_difference', 'score_difference', 'overall_difference']] = scaler.fit_transform(X.loc[:,['overall_home',
        'overall_away','home_score','away_score', 'weighted_points_home', 'rank_home', 'weighted_points_away', 
         'rank_away', 'rank_difference', 'average_rank','point_difference', 'score_difference', 'overall_difference']])
X.head() 

#Preprocessing - Data Value Changes - Encoding

# function that receives a specific feature and performs an ordinal encoding
def OrdinalEncoding(X,columnName):
    ft = OrdinalEncoder()
    ft.fit(X[columnName][:, np.newaxis])
    X[columnName] = ft.transform(X[columnName][:, np.newaxis]).reshape(-1)

# encode team name
OrdinalEncoding(X,'home_team')
OrdinalEncoding(X,'away_team')

# encode the country name
OrdinalEncoding(X,'country')

# encode win state
OrdinalEncoding(X,'is_won')
OrdinalEncoding(X,'won')

X.head(5) 

#Preprocessing - Feature Engineering - feature selection

# show a heatmap to see the correlation of each columns 
sns.heatmap(X.corr(),cmap = plt.cm.PuBu)
X.corr() 

# select the 3 most correlated features
X = X[['rank_difference', 'point_difference', 'overall_difference']]

# set the target feature
# target is a column indicating whether or not to win.
y = results[['won']]

# split into training and test data
# mix order, adjust distribution of labels using stratify
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, stratify=y)

# function that performs kfold cross-validation and returns accuracy
def kfoldAcc(k,X,y,model):
    kfold = KFold(n_splits=k)
    kf = cross_val_score(model, X, y.values.ravel(), cv=5)
    return kf

# Machine Learning Algorithm 1 - Decision Tree

# decisionTree for preliminary match, and visualize
# tree1
tree1 = DecisionTreeClassifier(max_depth=4, criterion="entropy")
tree1 = tree1.fit(X_train, y_train) # fit
dot_data2 = export_graphviz(tree1,
                                 out_file=None,
                                 feature_names=['rank_difference', 'point_difference', 'overall_difference'
                                                ],  # feature
                                 class_names=['Lose', 'Draw', 'Won'],  # target
                                 filled=True,
                                 rounded=True,
                                 special_characters=True)

graph2 = graphviz.Source(dot_data2)
graph = pydotplus.graph_from_dot_data(dot_data2)
graph.write_jpeg('decisiontree1.jpeg')

#confusion matrix for decision tree1
tree_1= tree1.predict(X_test)
print(confusion_matrix(y_test, tree_1))
print("\n")
# classification report
print(classification_report(y_test, tree_1))
# accuracy of k flod cross validation
print("Kfold accuarcy = ", kfoldAcc(5,X,y,tree1)) 

# tournaments decision tree
y_2 = results[['is_won']] # target
X_2 = X[['rank_difference', 'point_difference', 'overall_difference']] # features
# train_test split
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_2, y_2, test_size=0.1, shuffle=True, stratify=y)
# tree 2
tree2 = DecisionTreeClassifier(max_depth=4, criterion="entropy")
tree2 = tree2.fit(X_train_1, y_train_1)
dot_data3 = export_graphviz(tree2,
                                 out_file=None,
                                 feature_names=['rank_difference', 'point_difference', 'overall_difference',
                                                ],  # feature
                                 class_names=['Lose', 'Won'],  # target
                                 filled=True,
                                 rounded=True,
                                 special_characters=True)
graph2 = graphviz.Source(dot_data3)
graph = pydotplus.graph_from_dot_data(dot_data3)
graph.write_jpeg('decisiontree2.jpeg') 

# function that return the most optimal parameters with using GridSearchCV
def ensemble(params, clf):
    clf.fit(X_train, y_train)
    predict1 = clf.predict(X_test)
    grid_cv_1 = GridSearchCV(clf, param_grid=params, scoring="accuracy", n_jobs=-1, verbose=1)
    grid_cv_1.fit(X_train, y_train.values.ravel())
    return grid_cv_1

# Machine Learning Algorithm 2 - RandomForest

# Ensemble Learning (bagging and boosting)
# bagging (random forest)
# using gridSearch to search best parameters
params = {'n_estimators': [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
          'max_depth': [3,4,5,6]
          }
clf = RandomForestClassifier(random_state=0)
grid_cv_1 = ensemble(params, clf)
print('optimum hyper parameter(bagging): ', grid_cv_1.best_params_)
print('optimum predict accuracy(bagging):', kfoldAcc(5,X,y, grid_cv_1)) 

# predict 2022 world cup winner using decision tree
# let's define the rankings at the time of the World Cup
world_cup_rankings = rankings.loc[(rankings['rank_date'] == rankings['rank_date'].max()) &
                                  rankings['country_full'].isin(world_cup.index.unique())]

# Overall by country

world_cup_rankings["overall"] = [84.733333, 84.133333, 85.622222, 76.733333, 72.666667, 69.377778, 81.044444, 79.355556,
                                 76.666667, 85.511111, 86.555556, 85.177778, 76.688889, 71.638889, 76.022222, 79.644444,
                                 79.155556, 83.733333, 73.666667, 79.355556, 84.577778, 69.977778,
                                 69.977778, 79.844444, 80.888889, 75.466667, 85.266667, 78.622222, 71.888889, 75.644444,
                                 81.844444, 74.977778]

# world_cup_rankings = world_cup_rankings.fillna(9)
world_cup_rankings = world_cup_rankings.set_index(['country_full'])

from itertools import combinations

# Group qualifying round * 3
opponents = ['First match \nagainst', 'Second match\n against', 'Third match\n against']

# point
world_cup['points'] = 0

##Group qualifying round
for group in set(world_cup['Group']):
    print('___Starting group {}:___'.format(group))
    for home, away in combinations(world_cup.query('Group == "{}"'.format(group)).index, 2):
        print("{} vs. {}: ".format(home, away), end='')
        row = X[['rank_difference', 'point_difference', 'overall_difference']]
        home_rank = world_cup_rankings.loc[home, 'rank']
        home_points = world_cup_rankings.loc[home, 'weighted_points']
        home_overall = world_cup_rankings.loc[home, 'overall']
        opp_rank = world_cup_rankings.loc[away, 'rank']
        opp_points = world_cup_rankings.loc[away, 'weighted_points']
        opp_overall = world_cup_rankings.loc[away, 'overall']

        row['rank_difference'] = home_rank - opp_rank
        row['point_difference'] = home_points - opp_points
        row['overall_difference'] = home_overall - opp_overall

        # Predict with Decision Tree 1 (for Group qualifying round)
        Y_predicted = tree1.predict(row)
        points = 0
        if Y_predicted[0] == 'Win':
            print(home, " win")
            points = 3
            world_cup.loc[home, 'points'] += 3
        elif Y_predicted[0] == 'Lose':
            print(away, "win")
            world_cup.loc[away, 'points'] += 3
        else:
            print("Draw")
            world_cup.loc[home, 'points'] += 1
            world_cup.loc[away, 'points'] += 1

# Round of 16 matches
# [A_1th, B_2nd,C_1th, D_2nd,E_1th, F_2nd,G_1th, H_2nd,A_2nd, B_1th,C_2nd, D_1th,E_2nd, F_1th,G_2nd, H_1th ]
pairing = [0, 3, 4, 7, 8, 11, 12, 15, 1, 2, 5, 6, 9, 10, 13, 14]

world_cup = world_cup.sort_values(by=['Group', 'points'], ascending=False).reset_index()
# select the top 2
next_round_wc = world_cup.groupby('Group').nth([0, 1])
next_round_wc = next_round_wc.reset_index()
next_round_wc = next_round_wc.loc[pairing]
next_round_wc = next_round_wc.set_index('Team')
print(next_round_wc)

# tournament
finals = ['round_of_16', 'quarterfinal', 'semifinal', 'final']

for f in finals:
    print("___Starting of the {}___".format(f))
    iterations = int(len(next_round_wc) / 2)
    winners = []

    for i in range(iterations):
        home = next_round_wc.index[i * 2]
        away = next_round_wc.index[i * 2 + 1]
        print("{} vs. {}: ".format(home, away), end='')

        row = X[['rank_difference', 'point_difference', 'overall_difference']]
        home_rank = world_cup_rankings.loc[home, 'rank']
        home_points = world_cup_rankings.loc[home, 'weighted_points']
        home_overall = world_cup_rankings.loc[home, 'overall']
        opp_rank = world_cup_rankings.loc[away, 'rank']
        opp_points = world_cup_rankings.loc[away, 'weighted_points']
        opp_overall = world_cup_rankings.loc[away, 'overall']

        row['rank_difference'] = home_rank - opp_rank
        row['point_difference'] = home_points - opp_points
        row['overall_difference'] = home_overall - opp_overall

        # Predict with Decision Tree 2 (for tournament)
        Y_predicted2 = tree2.predict(row)
        if Y_predicted2[0] == True:
            print(home, "win")
            winners.append(home)
        elif Y_predicted2[0] == False:
            winners.append(away)
            print(away, "win")
    #The winning team
    next_round_wc = next_round_wc.loc[winners]
    print("\n") 

# predict 2022 world cup winner using MultiOutputRegressor(RandomForestClassifier())
# make a new dataset with required features to train the machine learning model
# Year,Country,Team_1,team_2, team_1_rank, team_2_rank, team_1_overall, team_2_overall,team_1_score, team_2_score

New_Dataset_part_1 = pd.DataFrame(list(zip(years,main.values[:,7],main.values[:,1],main.values[:,2], main.values[:,11], main.values[:,16], 
                                         main.values[:,21], main.values[:,23], main.values[:,3],main.values[:,4])),
                                columns=['year','Country','team_1','team_2','team_1_rank', 'team_2_rank', 'team_1_overall', 'team_2_overall','team_1_score','team_2_score'])

# Make a new dataset by changing the team_1 and team_2 and their respective scores
New_Dataset_part_2 = pd.DataFrame(list(zip(years,main.values[:,7],main.values[:,2],main.values[:,1], main.values[:,16], main.values[:,11], 
                                         main.values[:,23], main.values[:,21],main.values[:,4],main.values[:,3])),
                                columns=['year','Country','team_1','team_2', 'team_1_rank', 'team_2_rank', 'team_1_overall', 'team_2_overall','team_1_score','team_2_score'])

New_Dataset = pd.concat([New_Dataset_part_1,New_Dataset_part_2],axis=0)
New_Dataset = New_Dataset.sample(frac=1).reset_index(drop=True) # Shaffle the dataset
New_Dataset.head() 

# Definine the features and labels
Y= New_Dataset.iloc[:,8:10] # training targets (team_1_score,team_2_score, team_1_rank, team_2_rank, team_1_overall, team_2_overall)
categorized_data=New_Dataset.iloc[:,2:8].copy() # traing features

# label encoding instance
label_encoder = LabelEncoder()

# creating a list containg all the names of the countries
teams_1=New_Dataset.team_1.unique()
contries=New_Dataset.Country.unique()
all_countries=np.unique(np.concatenate((teams_1,contries), axis=0))
len(all_countries) 

# labeling the data using LabelEncorder in Sklearn-(Giving a unique number to each string(country))
label_encoder.fit(all_countries)

# list(label_encoder.classes_)
categorized_data['team_1']=label_encoder.transform(categorized_data['team_1'])
categorized_data['team_2']=label_encoder.transform(categorized_data['team_2'])

# convert these feature columns to categrize form to make the training processs more smoother
categorized_data['team_1']=categorized_data['team_1'].astype("category")
categorized_data['team_2']=categorized_data['team_2'].astype("category")

# Machine Learning Algorithm 3 - RandomForest

# make the model

# features
X = categorized_data

# MultiOutputRegressor
# multi target is predicted by approaching each one as a single target
# parameters: estimator
# need to predict scores for both countries

# RandomForestClassifier
# randomly learns a number of decision trees configured during the training and uses them for classification or regression results
model = MultiOutputRegressor(RandomForestClassifier())
model.fit(X,Y)

# making the predictions
prd=model.predict(X)

# create the confusion matrix for each predictions
score_team_1=[i[0] for i in prd]
score_team_2=[i[1] for i in prd]

cm1=confusion_matrix(list(Y.iloc[:,0]),score_team_1)
cm2=confusion_matrix(list(Y.iloc[:,1]),score_team_2)

# plotting the confussion matrix for score of team 01
plt.figure(figsize=(14,10))
sns.heatmap(cm1, annot=True,fmt="d", cmap='YlGnBu', linecolor='black', linewidths=1)
plt.title("Confusion Matrix for Team 1 Score")
plt.xlabel("Actual")
plt.ylabel("Predicted") 

# classification report to team 1 Score
# good Accuracy in predicting the team 1 Score
report_1=classification_report(Y.iloc[:,0],score_team_1)
print(report_1) 

# plotting the confussion matrix for score of team 02
plt.figure(figsize=(14,10))
sns.heatmap(cm2, annot=True,fmt="d", cmap='YlGnBu', linecolor='black', linewidths=1)
plt.title("Confusion Matrix for team 2 score")
plt.xlabel("Actual")
plt.ylabel("Predicted") 

# classification report to team 2 Score
# good accuracy in predicting the team 1 Score
report_2=classification_report(Y.iloc[:,1],score_team_2)
print(report_2) 

# fuction to select the winning team for the prediction array
def select_winning_team(probability_array):
    prob_lst=[round(probability_array[0][i],3) for i in range(2)]
    if (prob_lst[0]>prob_lst[1]):
        out=0
    elif (prob_lst[0]<prob_lst[1]):
        out=1
    elif (prob_lst[0]==prob_lst[1]):
        out=2
    return out,prob_lst

# 2022 FIFA Qatar World Cup Group Qualifying Countries and Groups

Group_A= ["Qatar","Ecuador","Senegal","Netherlands"]
Group_B= ["England","Iran","United States","Wales"]
Group_C= ["Argentina","Saudi Arabia","Mexico","Poland"]
Group_D= ["France","Denmark","Tunisia","Peru"]
Group_E= ["Spain", "Costa Rica","Germany","Japan"]
Group_F= ["Belgium", "Canada","Morocco","Croatia"]
Group_G= ["Brazil", "Serbia","Switzerland","Cameroon"]
Group_H= ["Portugal","Ghana","Uruguay","South Korea"]

Groups={"Group A":Group_A,"Group B":Group_B,"Group C":Group_C,"Group D":Group_D,
        "Group E":Group_E,"Group F":Group_F,"Group G":Group_G,"Group H":Group_H}

# group stage matches
# each team will play a league game to calculate points based on wins, draws, and losses

Group_standings={}
for grp_name in list(Groups.keys()):

    print(f"{grp_name} Matches")
    probable_countries = Groups[grp_name] # against country
    team_wins_dct = {}
    goal_scored_dct = {}
    goal_against_dct = {}
    win_dct = {} # win
    draw_dct = {} # draw
    lost_dct = {} # lost

    for i in range(len(probable_countries)):
        j=i+1
        team_1=probable_countries[i]
        team_1_num=label_encoder.transform([team_1])[0]
        team_wins=0

        # find the latest overalls and ranks of team1
        rank_1 = results[results['home_team'] == team_1]['rank_home']
        rank_1 = rank_1.iloc[-1].astype('int64')
        overall_1 = results[results['home_team'] == team_1]['overall_home']
        overall_1 = overall_1.iloc[-1].astype('int64')

        # team1 vs team 2
        # loop for team2
        while j<len((probable_countries)):

            team_2 = probable_countries[j] # against country
            team_2_num = label_encoder.transform([team_2])[0] # label encoding
            team_lst = [team_1,team_2]

            # find the latest overalls and ranks of team2
            rank_2 = results[results['home_team'] == team_2]['rank_home']
            rank_2 = rank_2.iloc[-1].astype('int64')
            overall_2 = results[results['home_team'] == team_2]['overall_home']
            overall_2 = overall_2.iloc[-1].astype('int64')

            # features
            Input_vector = np.array([[team_1_num,team_2_num, rank_1, rank_2, overall_1, overall_2]])
            res=model.predict(Input_vector) # predict

            win,prob_lst = select_winning_team(res)
            goal_scored_dct[team_1] = goal_scored_dct.get(team_1,0)+prob_lst[0]
            goal_scored_dct[team_2] = goal_scored_dct.get(team_2,0)+prob_lst[1]

            goal_against_dct[team_1] = goal_against_dct.get(team_1,0)+prob_lst[1]
            goal_against_dct[team_2] = goal_against_dct.get(team_2,0)+prob_lst[0]

            try:
                print(f" {team_1} vs {team_2} \nResults of the Match {res[0]}\n{team_lst[win]} wins \n")
                if (win) == 0:
                    team_wins_dct[team_1] = team_wins_dct.get(team_1,0) + 2 # get the point of team1 and plus 2
                    team_wins_dct[team_2] = team_wins_dct.get(team_2,0)
                    
                    win_dct[team_1] = win_dct.get(team_1,0)+1 # team1 + 1 win point
                    win_dct[team_2] = win_dct.get(team_2,0)
                    lost_dct[team_2] = lost_dct.get(team_2,0)+1 # team2 + 1 lost point
                    lost_dct[team_1] = lost_dct.get(team_1,0)
                    draw_dct[team_2] = draw_dct.get(team_2,0)
                    draw_dct[team_1] = draw_dct.get(team_1,0)

                elif (win) == 1:
                    team_wins_dct[team_2] = team_wins_dct.get(team_2,0)+2 # get the point of team2 and plus 2
                    team_wins_dct[team_1] = team_wins_dct.get(team_1,0)
                    
                    win_dct[team_2] = win_dct.get(team_2,0) + 1 # team2 + 1 win point
                    win_dct[team_1] = win_dct.get(team_1,0)
                    lost_dct[team_1] = lost_dct.get(team_1,0) + 1 # team1 + 1 lost point
                    lost_dct[team_2] = lost_dct.get(team_2,0)
                    draw_dct[team_1] = draw_dct.get(team_1,0)
                    draw_dct[team_2] = draw_dct.get(team_2,0)

            except IndexError: # Draw
                print(f"{team_1} vs {team_2} \nResults of the Match {res[0]}\nMatch Draw\n") 
                team_wins_dct[team_1] = team_wins_dct.get(team_1,0)+1 # get the point of team1 and plus 1
                team_wins_dct[team_2] = team_wins_dct.get(team_2,0)+1 # get the point of team2 and plus 1
                
                draw_dct[team_1] = draw_dct.get(team_1,0)+1 # team1 + 1 win point
                draw_dct[team_2] = draw_dct.get(team_2,0)+1 # team2 + 1 win point
                
                win_dct[team_1] = win_dct.get(team_1,0)
                lost_dct[team_1] = lost_dct.get(team_1,0)
                
                win_dct[team_2] = win_dct.get(team_2,0)
                lost_dct[team_2] = lost_dct.get(team_2,0)
                    
            j=j+1

    group_results = [win_dct,draw_dct,lost_dct,team_wins_dct,goal_scored_dct,goal_against_dct]
    Group_standings[grp_name] = group_results 

# visualization of group qualification

for grp_name in list(Group_standings.keys()):

    team_wins_dct= dict(sorted(Group_standings[grp_name][3].items()))
    goal_scored_dct=dict(sorted(Group_standings[grp_name][4].items()))
    goal_against_dct=dict(sorted(Group_standings[grp_name][5].items()))
    
    win_dct=dict(sorted(Group_standings[grp_name][0].items()))
    draw_dct=dict(sorted(Group_standings[grp_name][1].items()))
    lost_dct=dict(sorted(Group_standings[grp_name][2].items()))
    
    lst_teams=list(team_wins_dct.keys())
    
    win_lst=list(win_dct.values())
    draw_lst=list(draw_dct.values())
    lost_lst=list(lost_dct.values())
    
    lst_win_count=list(team_wins_dct.values())
    goal_scored=list(goal_scored_dct.values())
    goal_against=list(goal_against_dct.values())
    goal_differance=[goal_scored[i]-goal_against[i] for i in range (len(goal_scored))]
    ranking_table=pd.DataFrame(list(zip(lst_teams,win_lst,draw_lst,lost_lst,goal_scored,goal_against,goal_differance,lst_win_count)),
                               columns=["Team","Wins","Draw","Lost","Goal Scored","Goal Against","Goal Differance","Points"])
    ranking_table=ranking_table.sort_values("Points",ascending=False).reset_index(drop=True)
    ranking_table.index = ranking_table.index + 1
    print(f"\n\n{grp_name} Final Rankings")
    print(ranking_table.to_markdown()) 

# Round of 16 section 1 and 2
# Same structure as group stage

# Round of 16 section 1

qualified_teams_1=[]
standings=list(Group_standings.keys())
i=0

print(f"--- Round of 16 ---\n")

while i < (len(standings)):

    A_team= sorted(Group_standings[standings[i]][3].items(), key=lambda x: x[1], reverse=True)
    team_1=A_team[0][0]
    B_team= sorted(Group_standings[standings[i+1]][3].items(), key=lambda x: x[1], reverse=True)
    team_2=B_team[1][0]
    
    team_1_num=label_encoder.transform([team_1])[0]
    team_2_num=label_encoder.transform([team_2])[0]
    team_lst=[team_1,team_2]

    rank_1 = results[results['home_team'] == team_1]['rank_home']
    rank_1 = rank_1.iloc[-1].astype('int64')
    rank_2 = results[results['home_team'] == team_2]['rank_home']
    rank_2 = rank_2.iloc[-1].astype('int64')

    overall_1 = results[results['home_team'] == team_1]['overall_home']
    overall_1 = overall_1.iloc[-1].astype('int64')
    overall_2 = results[results['home_team'] == team_2]['overall_home']
    overall_2 = overall_2.iloc[-1].astype('int64')
    
    Input_vector=np.array([[team_1_num,team_2_num, rank_1, rank_2, overall_1, overall_2]])
    res=model.predict(Input_vector)
    win,_=select_winning_team(res)

    try:
            print(f"{team_1} vs {team_2} \n{team_lst[win]} wins ")
            print(f"{team_lst[win]} into the Quater-Finals\n")
            qualified_teams_1.append(team_lst[win])

    except IndexError:
            print(f"{team_1} vs {team_2} \nMatch Draw") 
            winning_team=random.choice(team_lst)
            print(f"{winning_team} wins at Penaly Shoot-Out")
            print(f"{winning_team} into the Quater-Finals \n")
            qualified_teams_1.append(winning_team)

    i=i+2
    

# Round of 16 section 2

qualified_teams_2=[]
standings=list(Group_standings.keys())
i=0

while i < (len(standings)):

    A_team= sorted(Group_standings[standings[i]][3].items(), key=lambda x: x[1], reverse=True)
    team_1=A_team[1][0]
    B_team= sorted(Group_standings[standings[i+1]][3].items(), key=lambda x: x[1], reverse=True)
    team_2=B_team[0][0]
    
    team_1_num=label_encoder.transform([team_1])[0]
    team_2_num=label_encoder.transform([team_2])[0]
    team_lst=[team_1,team_2]
    
    rank_1 = results[results['home_team'] == team_1]['rank_home']
    rank_1 = rank_1.iloc[-1].astype('int64')
    rank_2 = results[results['home_team'] == team_2]['rank_home']
    rank_2 = rank_2.iloc[-1].astype('int64')

    overall_1 = results[results['home_team'] == team_1]['overall_home']
    overall_1 = overall_1.iloc[-1].astype('int64')
    overall_2 = results[results['home_team'] == team_2]['overall_home']
    overall_2 = overall_2.iloc[-1].astype('int64')
    
    Input_vector=np.array([[team_1_num,team_2_num, rank_1, rank_2, overall_1, overall_2]])
    res=model.predict(Input_vector)
    win,_=select_winning_team(res)

    try:
            print(f"{team_1} vs {team_2} \n{team_lst[win]} wins ")
            print(f"{team_lst[win]} into the Quater-Finals \n")
            qualified_teams_2.append(team_lst[win])
            
    except IndexError:
            print(f"{team_1} vs {team_2} \nMatch Draw") 
            winning_team=random.choice(team_lst)
            print(f"{winning_team} wins at Penaly Shoot-Out")
            print(f" {winning_team} into the Quater-Finals \n")
            qualified_teams_2.append(winning_team)

    i=i+2 
print(f"Teams selected to the Quater Finals - {qualified_teams_1+qualified_teams_2}") 

# Quarter Finals

Semifinal_teams=[]
i=0
print(f"--- Quater Final Matches ---\n")

while i < (len(qualified_teams_1))-1:

    team_1= qualified_teams_1[i]
    team_2= qualified_teams_1[i+1]
    
    team_1_num=label_encoder.transform([team_1])[0]
    team_2_num=label_encoder.transform([team_2])[0]
    team_lst=[team_1,team_2]
    
    rank_1 = results[results['home_team'] == team_1]['rank_home']
    rank_1 = rank_1.iloc[-1].astype('int64')
    rank_2 = results[results['home_team'] == team_2]['rank_home']
    rank_2 = rank_2.iloc[-1].astype('int64')

    overall_1 = results[results['home_team'] == team_1]['overall_home']
    overall_1 = overall_1.iloc[-1].astype('int64')
    overall_2 = results[results['home_team'] == team_2]['overall_home']
    overall_2 = overall_2.iloc[-1].astype('int64')
    
    Input_vector=np.array([[team_1_num,team_2_num, rank_1, rank_2, overall_1, overall_2]])
    res=model.predict(Input_vector)
    win,_=select_winning_team(res)

    try:
            print(f"{team_1} vs {team_2} \n{team_lst[win]} wins")
            print(f"{team_lst[win]} into the Semi-Finals \n")
            Semifinal_teams.append(team_lst[win])
            
    except IndexError:
            print(f"{team_1} vs {team_2} \nMatch Draw ")
            winning_team=random.choice(team_lst)
            print(f"{winning_team} wins at Penaly Shoot-Out")
            print(f"{winning_team} into the Semi-Finals\n")
            Semifinal_teams.append(winning_team)

    i=i+2
    
i=0

while i < (len(qualified_teams_2))-1:

    team_1= qualified_teams_2[i]
    team_2= qualified_teams_2[i+1]
    team_1_num=label_encoder.transform([team_1])[0]
    team_2_num=label_encoder.transform([team_2])[0]
    team_lst=[team_1,team_2]
    
    rank_1 = results[results['home_team'] == team_1]['rank_home']
    rank_1 = rank_1.iloc[-1].astype('int64')
    rank_2 = results[results['home_team'] == team_2]['rank_home']
    rank_2 = rank_2.iloc[-1].astype('int64')

    overall_1 = results[results['home_team'] == team_1]['overall_home']
    overall_1 = overall_1.iloc[-1].astype('int64')
    overall_2 = results[results['home_team'] == team_2]['overall_home']
    overall_2 = overall_2.iloc[-1].astype('int64')
    
    Input_vector=np.array([[team_1_num,team_2_num, rank_1, rank_2, overall_1, overall_2]])
    res=model.predict(Input_vector)
    win,_=select_winning_team(res)

    try:
            print(f"{team_1} vs {team_2} \n{team_lst[win]} wins")
            print(f"{team_lst[win]} into the Semi-Finals\n")
            Semifinal_teams.append(team_lst[win])
            
    except IndexError:
            print(f"{team_1} vs {team_2} \nMatch Draw") 
            winning_team=random.choice(team_lst)
            print(f"{winning_team} wins at Penaly Shoot-Out")
            print(f"{winning_team} into the Semi-Finals\n")
            Semifinal_teams.append(winning_team)
            
    i=i+2 

print(f"Teams selected to the Semi-Finals - {Semifinal_teams}") 

#Semi Finals

final_teams=[]
third_place_match_teams=[]
i=0

print(f"--- Semi Final Matches ---\n")

while i < (len(Semifinal_teams))-1:

    team_1= Semifinal_teams[i]
    team_2= Semifinal_teams[i+1]
    
    team_1_num=label_encoder.transform([team_1])[0]
    team_2_num=label_encoder.transform([team_2])[0]
    team_lst=[team_1,team_2]
    
    rank_1 = results[results['home_team'] == team_1]['rank_home']
    rank_1 = rank_1.iloc[-1].astype('int64')
    rank_2 = results[results['home_team'] == team_2]['rank_home']
    rank_2 = rank_2.iloc[-1].astype('int64')

    overall_1 = results[results['home_team'] == team_1]['overall_home']
    overall_1 = overall_1.iloc[-1].astype('int64')
    overall_2 = results[results['home_team'] == team_2]['overall_home']
    overall_2 = overall_2.iloc[-1].astype('int64')
    
    Input_vector=np.array([[team_1_num,team_2_num, rank_1, rank_2, overall_1, overall_2]])
    res=model.predict(Input_vector)
    win,_=select_winning_team(res)

    try:
            print(f"{team_1} vs {team_2} \n{team_lst[win]} wins ")
            print(f"{team_lst[win]} into the FIFA-Finals \n")
            final_teams.append(team_lst[win])
            third_place_match_teams.append(team_lst[(win+1)%2])

            
    except IndexError:
            print(f"{team_1} vs {team_2} \nMatch Draw") 
            winning_team=random.choice(team_lst)
            print(f"{winning_team} wins at Penaly Shoot-Out")
            print(f"{winning_team} into the FIFA-Finals\n")
            final_teams.append(winning_team)
            team_lst.remove(winning_team)
            third_place_match_teams.append(team_lst[0])

    i=i+2 

print(f"Teams selected to the Finals - {final_teams}")
print(f"Teams selected to the Third Place match - {third_place_match_teams}") 

# Finals and Third Place match

print(f"--- Final Match ---\n")
team_1= final_teams[1]
team_2= final_teams[0]
    
team_1_num=label_encoder.transform([team_1])[0]
team_2_num=label_encoder.transform([team_2])[0]
team_lst=[team_1,team_2]
    
rank_1 = results[results['home_team'] == team_1]['rank_home']
rank_1 = rank_1.iloc[-1].astype('int64')
rank_2 = results[results['home_team'] == team_2]['rank_home']
rank_2 = rank_2.iloc[-1].astype('int64')

overall_1 = results[results['home_team'] == team_1]['overall_home']
overall_1 = overall_1.iloc[-1].astype('int64')
overall_2 = results[results['home_team'] == team_2]['overall_home']
overall_2 = overall_2.iloc[-1].astype('int64')
    
Input_vector=np.array([[team_1_num,team_2_num, rank_1, rank_2, overall_1, overall_2]])
res=model.predict(Input_vector)
win,_=select_winning_team(res)

try:
    print(f"{team_1} vs {team_2} \n{team_lst[win]} are the Winners\n\n")
    winner=team_lst[win]
    place_2=team_lst[(win+1)%2]
            
except IndexError:
    print(f"{team_1} vs {team_2} \nMatch Draw") 
    winning_team=random.choice(team_lst)
    print(f"{winning_team} wins at Penaly Shoot-Out")
    print(f"{winning_team} are the Winners\n\n")
    winner=winning_team
    
    team_lst.remove(winning_team)
    place_2=team_lst[0]

print(f"Third Place match\n")
team_1= third_place_match_teams[1]
team_2= third_place_match_teams[0]
    
team_1_num=label_encoder.transform([team_1])[0]
team_2_num=label_encoder.transform([team_2])[0]
team_lst=[team_1,team_2]
    
rank_1 = results[results['home_team'] == team_1]['rank_home']
rank_1 = rank_1.iloc[-1].astype('int64')
rank_2 = results[results['home_team'] == team_2]['rank_home']
rank_2 = rank_2.iloc[-1].astype('int64')

overall_1 = results[results['home_team'] == team_1]['overall_home']
overall_1 = overall_1.iloc[-1].astype('int64')
overall_2 = results[results['home_team'] == team_2]['overall_home']
overall_2 = overall_2.iloc[-1].astype('int64')
    
Input_vector=np.array([[team_1_num,team_2_num, rank_1, rank_2, overall_1, overall_2]])
res=model.predict(Input_vector)
win,_=select_winning_team(res)

try:
    print(f"{team_1} vs {team_2} \n  {team_lst[win]} Wins the 3rd Place \n")
    place_3=team_lst[win]
            
except IndexError:
    print(f"{team_1} vs {team_2} \nMatch Draw ") 
    winning_team=random.choice(team_lst)
    print(f"{winning_team} wins at Penaly Shoot-Out ")
    print(f"{winning_team} Wins the 3rd Place \n")
    place_3=winning_team
    

print(f"-------- 1st Place is {winner} ----------")
print(f"-------- 2nd Place is {place_2} ----------")
print(f"-------- 3rd Place is {place_3} ----------") 

# Machine Learning Algorithm 4 - Linear Regression

# use linear regression to see how overall and rank differences affect score differences

X = results.copy()

x = X[['overall_difference', 'rank_difference']]
y = X['score_difference']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

# create an object of LinearRegression class
model = LinearRegression()

# fit the training data
model.fit(x_train, y_train)

# print the score of linear regression
score = model.score(x_train, y_train)
score 

#linear regression plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("overall", size = 14, color = "r")
ax.set_ylabel("ranking", size = 14, color = "r")
ax.set_zlabel("score", size = 14, color = "r")
ax.scatter(X['overall_difference'], X['rank_difference'], score, c='green',marker='o', s=15, cmap='Greens') 

# Machine Learning Algorithm 5 - Clustering

# check the power of countries by creating clusters based on ranks and overalls by country

X = results.copy()

X = X[['home_team', 'rank_home', 'overall_home']] # features
X = X.drop_duplicates('home_team') # remove duplicate country names
X = X.drop('home_team', axis = 1)

model = KMeans(n_clusters=4)
k = 3 # k can be changed
model = KMeans(n_clusters=k)
model.fit(X) #fit
# show clusters with scatter plot
plt.scatter(X['overall_home'], X['rank_home'], c=model.labels_.astype(float), s=50, alpha=0.5)
plt.show() 
 













<Source>
-code-
Kaggle - MANINKA123 <FIFA 2022(EDA + Prediction Model)>
https://www.kaggle.com/code/pasinduranasinghe123/fifa-2022-eda-prediction-model
Kaggle - MJEREMY < World Cup Winner Prediction 2018 >
https://www.kaggle.com/code/zhangyue199/world-cup-winner-prediction-2018

-Dataset-
Kaggle - MART JÜRISOO <International football results from 1872 to 2022>
https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
Kaggle - TADHG FITZGERALD < FIFA Soccer Rankings>
https://www.kaggle.com/datasets/tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now
Kaggle - STEFANO LEONE < FIFA 21 complete player dataset>
https://www.kaggle.com/datasets/stefanoleone992/fifa-21-complete-player-dataset

<Teamwork data>
 
 
<Output >
   
   
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
 
 
  
 
  
 
  
 
  
 
  
 
  
 
 
  
 
 
  
 
 
  
 
 
  
 
  
 
  
 
 
 
  
 
  
 
  
 
  
 
  
 
  
 
  
    
  
 
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 
  
 

