from tsai.basics import *

import random
import numpy as np
from sklearn.metrics import mean_squared_error
from  sklearn.metrics import r2_score

import pandas as pd 
import gma


filename = r"train.csv"
data = pd.read_csv(filename, header=0)


PRS = data['Press']   #hpa
WIN = data['WindS']   #m/s
TMAX = data['Ta_Max']   #℃
TMIN = data['Ta_Min']   #℃
RHU = data['RH']      #%
LAT = data['Lat']     #°
Day = data['DOY']
ELE = data['DEM']     #m

prs = PRS.values
win = WIN.values
tmax = TMAX.values
tmin = TMIN.values
rh = RHU.values
lat = LAT.values[1]
doy = Day.values
ele = ELE.values[1]
ssd = gma.climet.Other.DaylightHours(doy,lat)


et0 = gma.climet.ET0.PenmanMonteith(prs, win, tmax, tmin, rh, ssd, lat, doy, ele)


data['ET0'] = et0
#filename1 = filename.replace('.','_et0.')
#data.to_csv(filename1, index=False, header=True)



num_step=1
num_feature=53
    

data_raw_t = data
data_raw_X = data_raw_t.drop(['name','date','ET','ET0'], axis=1)
#data_raw_X = data_raw_t.drop(['name','date','ET'], axis=1)
data_raw_y = data['ET']

X_raw = data_raw_X.values
y_raw = data_raw_y.values

N = len(data)
num_samples = int(N/num_step)
X = X_raw.reshape(num_samples, -1, num_step)
y = y_raw.reshape(num_samples, -1)




splits = get_splits(X, valid_size=0.25, shuffle=True, balance=False,
                    train_only=False, check_splits=True, random_state=165, show_plot=True, verbose=True)



et0_raw = data['ET0'].values
et0 = et0_raw.reshape(num_samples, -1)
et0_test = et0[splits[1]]
#et0_test = et0[splits[1]][:,-1]




import optuna
from optuna.integration import FastAIPruningCallback
from optuna.samplers import TPESampler


tfms = [None, TSRegression()]
batch_tfms = TSStandardize(by_var=True)


def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.modules():
        if type(module) is nn.Conv2d:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)



def objective(trial:optuna.Trial):
    
    # Define search space
    d_model = trial.suggest_categorical('d_model', [64, 128, 256])
    n_heads = trial.suggest_categorical('n_heads', [4, 8, 16, 32])
    d_ff = trial.suggest_categorical('d_ff', [256, 512])
    dropout = trial.suggest_uniform("dropout", 0.0, 0.5) 
    n_layers = trial.suggest_int('n_layers', 2, 5, step=1)
    fc_dropout = trial.suggest_uniform("fc_dropout", 0.0, 0.8)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    reg_lambda = trial.suggest_uniform("reg_lambda", 0.0, 1.0)
    reg_beta1 =  trial.suggest_uniform("reg_beta1", 0.0, 1.0)
    reg_beta2 =  trial.suggest_uniform("reg_beta2", 0.0, 1.0)
    
    learn = TSRegressor(X, y, splits=splits, path='models', arch="TSTPlus", 
                        arch_config={'d_model':d_model, 'n_heads':n_heads
                                     , 'd_ff':d_ff
                                     , 'dropout':dropout,'n_layers':n_layers
                                     , 'fc_dropout':fc_dropout
                                     },
                        tfms=tfms, 
                        batch_tfms=batch_tfms, metrics=rmse, cbs=FastAIPruningCallback(trial), verbose=True, seed=100)


    with ContextManagers([learn.no_logging(), learn.no_bar()]): 
        learn.fit_one_cycle(5, lr_max=learning_rate)

    # Return the objective value
    los = float(learn.recorder.values[-1][1]) # return the validation loss value of the last epoch 
    #y_test_probas, _, y_test_preds = learn.get_X_preds(X[splits[1]])
    #phy = np.sqrt(mean_squared_error(et0_test, y_test_preds))
    
    Rw = l2_regularization(learn,reg_lambda)
    
    y_test_probas, _, y_preds = learn.get_X_preds(X)
    #phy = np.sqrt(mean_squared_error(et0, y_preds))
    phy1 = np.mean(torch.relu(torch.tensor(y_preds*(et0-y_preds))).numpy())
    
    phy_sum = 0
    y_pred_phy = np.array(y_preds)
    for i in range(1,N):
        phy_num = torch.relu(torch.tensor((y_pred_phy[i]-y_pred_phy[i-1])*(et0[i]-et0[i-1]))).numpy()
        phy_sum += phy_num
    print (phy_sum)
    phy2 = phy_sum/(N-1)
    
    obj = los + Rw + reg_beta1*phy1 + reg_beta2*phy2
    return obj
    #return los
#study.trials[0].intermediate_values[4]

seed = 42
study = optuna.create_study(sampler=TPESampler(seed=seed), directions=['minimize'])
study.optimize(objective, n_trials=1000)

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    
display(optuna.visualization.plot_optimization_history(study))
display(optuna.visualization.plot_param_importances(study))
display(optuna.visualization.plot_slice(study))
display(optuna.visualization.plot_parallel_coordinate(study))


# Get the best nf and dropout rate from the best trial object
trial = study.best_trial
d_model = trial.params['d_model']
n_heads = trial.params['n_heads']
d_ff = trial.params['d_ff']
dropout = trial.params['dropout']
n_layers = trial.params['n_layers']
fc_dropout = trial.params['fc_dropout']
learning_rate = trial.params['learning_rate']

learn = TSRegressor(X, y, splits=splits, path='models', arch="TSTPlus", 
                        arch_config={'d_model':d_model, 'n_heads':n_heads
                                     , 'd_ff':d_ff
                                     , 'dropout':dropout,'n_layers':n_layers
                                     , 'fc_dropout':fc_dropout
                                     },
                        tfms=tfms, 
                        batch_tfms=batch_tfms, metrics=rmse, cbs=ShowGraph(), verbose=True, seed=seed)

learn.fit_one_cycle(100, lr_max=learning_rate)
#learn.export("model.pkl") 


#predict
_, _, preds = learn.get_X_preds(X)
y_pred = np.array(preds)

para = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
#para.to_csv(r'D:\data\data_Flux\GEE_data_new\site_features\feature_use\raw\output\para.csv')

#importance
from tsai.learner import ts_learner
colnames = data_raw_X.columns.tolist()
learn.feature_importance(feature_names=colnames,title='Feature Importance')
