import optuna
import plotly

# load the desired Optuna study
loaded_study = optuna.load_study(study_name='PPO_V36_tune', storage='mysql+pymysql://<user>:@<host>/<dbname>')
# show paramteres of all trials in this study
print(len(loaded_study.trials))
for each in loaded_study.trials:
    print(each)
print('Best trial :', loaded_study.best_trial)

# get variable importance
importance_dict = optuna.importance.get_param_importances(loaded_study)

# show variable importance in a graph
fig = optuna.visualization.plot_param_importances(loaded_study)
fig.show()
