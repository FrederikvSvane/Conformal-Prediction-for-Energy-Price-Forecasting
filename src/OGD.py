import numpy as np

class OGD_Predictor:
    def __init__(self, alpha=0.1, eta=0.5, q_init=1.0):
        """
            alpha: Target miscoverage rate
            eta: Learning rate
            q_init: Initial threshold
        """
        self.alpha = alpha
        self.eta = eta
        self.q = q_init
        
        self.coverage_history = []
        self.threshold_history = []
        
    def get_interval(self, y_pred):
        """
        Gets the interval where the spotprice will be with 1-alpha confidence
        """
        lower = y_pred - self.q
        upper = y_pred + self.q
        return lower, upper
    
    def update(self, y_pred, y_true):
        """
        Update threshold after seeing true value
        """
        # Non-conformity score function
        def score_function(y_true, y_pred):
            return abs(y_true - y_pred)

        score = score_function(y_true, y_pred)
        
        covered = (score <= self.q)
        err_t = 0 if covered else 1
        
        # Update q
        self.q = self.q + self.eta * (err_t - self.alpha)
        self.q = max(self.q, 0.001)  # Keep positive
        
        self.coverage_history.append(covered)
        self.threshold_history.append(self.q)
        
        return covered
    
    def get_coverage_rate(self):
        """Get overall coverage rate"""
        if not self.coverage_history:
            return None
        return np.mean(self.coverage_history)