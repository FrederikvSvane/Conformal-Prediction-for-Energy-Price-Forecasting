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
import json

class SARIMAX_model:
    def __init__(self):
        self.fitted_model = None
        self.best_order = None
        self.best_seasonal_order = None
        
    def auto_train(self, y_train, X_train, 
                   start_p=0, max_p=3,
                   start_q=0, max_q=3,
                   seasonal=True, m=24,
                   start_P=0, max_P=1,
                   start_Q=0, max_Q=1,
                   max_D=1,
                   information_criterion='bic',
                   trace=True,
                   n_jobs=1):
        print("Running auto_arima to find best parameters...")
        
        auto_model = auto_arima(
            y_train,
            exogenous=X_train,
            start_p=start_p, max_p=max_p,
            start_q=start_q, max_q=max_q,
            seasonal=seasonal,
            m=m,
            start_P=start_P, max_P=max_P,
            start_Q=start_Q, max_Q=max_Q,
            max_D=max_D,
            trace=trace,
            information_criterion=information_criterion,
            n_jobs=n_jobs,
        )
        
        self.fitted_model = auto_model
        self.best_order = auto_model.order
        self.best_seasonal_order = auto_model.seasonal_order

        return self
    
    def fit(self, y_train, X_train, order=(1,1,0), seasonal_order=(1,0,0,24)):
        """
        Fit SARIMAX with specific parameters (no auto search)
        """
        print(f"Fitting SARIMAX with order={order}, seasonal_order={seasonal_order}")
        
        model = SARIMAX(
            y_train,
            exog=X_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.fitted_model = model.fit(disp=False)
        self.best_order = order
        self.best_seasonal_order = seasonal_order
        
        return self
    
    def predict(self, X_test, steps=None):
        """
        Make predictions on test data
        """
        if steps is None:
            steps = len(X_test)
        
        forecast = self.fitted_model.forecast(steps=steps, exog=X_test)
        return forecast
    
    def save(self, path):
        """Save the model parameters (not the full fitted model to reduce size)"""
        save_dict = {
            'best_order': self.best_order,
            'best_seasonal_order': self.best_seasonal_order
        }
        with open(path, 'w') as f:
            json.dump(save_dict, f)
        print(f"Model parameters saved to {path}")