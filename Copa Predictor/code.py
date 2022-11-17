import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

matches = pd.read_csv('international_matches.csv')
matches['home_team_code'] = matches['home_team'].astype('category').cat.codes
matches['away_team_code'] = matches['away_team'].astype('category').cat.codes
matches.to_csv('imputed_matches.csv', index=False)
numeric_cols = [col for col in matches.columns if matches[col].dtype != 'object']
imputer = SimpleImputer()
imputed_matches = pd.DataFrame(imputer.fit_transform(matches[numeric_cols]))
imputed_matches.columns = numeric_cols

y = imputed_matches.home_team_score
X = imputed_matches[['home_team_fifa_rank', 'away_team_fifa_rank', 'home_team_total_fifa_points', 'away_team_total_fifa_points', 'neutral_location', 'home_team_code', 'away_team_code']]
forest_model = RandomForestRegressor(random_state=12)
forest_model.fit(X, y)
X_pred = pd.read_csv('jogos.csv')
print(forest_model.predict(X_pred))


