from pickle import dump, load

import torch
import torch.nn as nn


class LSTM_model(nn.Module):
    # in: hour, day, week, year, month, day_of_year (sin/cos), wind, consumption, temp, normal temp, temp*consumption, temp_deviation, spot_lag1
    # out: spot price at t
    def __init__(self, input_size=14, hidden_size=64, num_layers=2, output_size=1):
        super(LSTM_model, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            batch_first=True) #Makes the input shape = (batch_size, sequence_length, input_size)
        
        self.linear = nn.Linear(in_features=hidden_size,
                               out_features=output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.linear(lstm_out[:, -1, :]) # lstm_out has shape (batch_size, sequence_length, hidden_size), so doing [:, -1, :] just says return entire batch, the LAST time step and all hidden features
        return output


from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from pickle import dump, load

class SARIMAX_model:
    # The model is just holding which parameters and the order to use
    def __init__(self):
        self.params = None
        self.best_order = None
    
    def auto_train(self, y_train, X_train, 
               start_p=0, max_p=5,
               start_q=0, max_q=5,
               n_jobs=1):
        
        auto_model = auto_arima(
            y_train,
            exogenous=X_train,
            start_p=start_p, max_p=max_p,
            start_q=start_q, max_q=max_q,
            seasonal=False,
            maxiter=200,
            suppress_warnings=True,
            trace=True,
            information_criterion='bic',
            n_jobs=n_jobs,
        )
        
        self.best_order = auto_model.order
        
        # Preffering to use statsmodels over pmdarima as we've had the best experiences with it. 
        # However, pmdarima has the auto_arima which is very useful for order selection. 
        # Alternatively, we could do a simple grid search with 'bic' ourselves, but this works just fine.
        model = SARIMAX(
            y_train,
            exog=X_train,
            order=self.best_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted = model.fit(disp=False)
        
        self.params = fitted.params
        
        return self
       
    def predict(self, y_train, X_train, steps=None, X_test=None):
        """
        Make predictions on test data
        """
        # Create model from inputs
        model = SARIMAX(
            y_train,
            exog=X_train,
            order=self.best_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        filtered = model.filter(self.params)
        forecast = filtered.forecast(steps=steps, exog=X_test)
        
        return forecast
    
    def save(self, path):
        save_dict = {
            'params': self.params,
            'order': self.best_order
        }
        with open(path, 'wb') as f:
            dump(save_dict, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            save_dict = load(f)
            self.params = save_dict['params']
            self.best_order = save_dict['order']
        return self