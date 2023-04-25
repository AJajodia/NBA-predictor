# import modules

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

filepath = "/Users/anu/PycharmProjects/NBA-predictor/NBA_Team_Stats.csv"

team_data = pd.read_csv(filepath)

filepath2 = "/Users/anu/PycharmProjects/NBA-predictor/2023.csv"

present_data = pd.read_csv(filepath2)

# print(team_data.describe())
# print(team_data.head())
# print(list(team_data))

features = ['Points', 'Assists', 'Steals', 'Blocks', 'Turnovers', 'Personal Fouls',
            'Drebounds', 'Orebounds', 'fg-pct', '3-pct', 'ft-pct']

y = team_data.No

X = team_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

nba_model = RandomForestRegressor(random_state=3)

nba_model.fit(train_X, train_y)

print(mean_absolute_error(val_y, nba_model.predict(val_X)))

values = nba_model.predict(present_data)
teams = ['Milwaukee Bucks', 'Boston Celtics', 'Philadelphia 76ers', 'Denver Nuggets', 'Cleveland Cavaliers', 'Memphis Grizzlies', 'Sacramento Kings', 'New York Knicks', 'Brooklyn Nets', 'Phoenix Suns', 'Golden State Warriors', 'LA Clippers', 'Miami Heat', 'Los Angeles Lakers', 'Minnesota Timberwolves', 'New Orleans Pelicans', 'Atlanta Hawks', 'Toronto Raptors', 'Chicago Bulls', 'Oklahoma City Thunder', 'Dallas Mavericks', 'Utah Jazz', 'Indiana Pacers', 'Washington Wizards', 'Orlando Magic', 'Portland Trail Blazers', 'Charlotte Hornets', 'Houston Rockets', 'San Antonio Spurs', 'Detroit Pistons']
scores = []
print(len(values))
for i in range(len(values)):
    scores.append((teams[i], values[i]))
scores = sorted(scores, key=lambda x: x[1])
for i in range(len(scores)):
    print(str(i+1)+" " + scores[i][0])
