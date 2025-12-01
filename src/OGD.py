import numpy as np

class OGD_Predictor:
    def __init__(self, alpha=0.1, eta=0.5, q_init=1.0):
        self.alpha = alpha
        self.eta = eta
        self.q = q_init
        self.coverage_history = []
        self.threshold_history = []
    
    def get_interval(self, y_pred):
        lower = y_pred - self.q
        upper = y_pred + self.q
        return lower, upper
    
    def _compute_update_term(self, y_pred, y_true):
        """Compute the update term for threshold adjustment"""
        score = abs(y_true - y_pred)
        covered = (score <= self.q)
        err_t = 0 if covered else 1
        
        # OGD: just uses (err_t - alpha)
        return err_t - self.alpha
    
    def update(self, y_pred, y_true):
        score = abs(y_true - y_pred)
        covered = (score <= self.q)
        
        update_term = self._compute_update_term(y_pred, y_true)
        
        # Update q_t
        self.q = self.q + self.eta * update_term
        self.q = max(self.q, 0.001)
        
        self.coverage_history.append(covered)
        self.threshold_history.append(self.q)
        
        return covered
    
    def get_coverage_rate(self):
        if not self.coverage_history:
            return None
        return np.mean(self.coverage_history)

# Inhereted class of OGD_Predictor, which just makes it own _compute_update_term and uses the parent class's update and get_interval
class ECI_Predictor(OGD_Predictor):
    def __init__(self, alpha=0.1, eta=0.5, q_init=1.0, c=1.0, version='basic', 
                eq_function='sigmoid'):
        super().__init__(alpha, eta, q_init)
        self.c = c
        self.version = version
        self.eq_function = eq_function  # 'sigmoid' or 'gaussian'

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-self.c * x))
    
    def _sigmoid_grad(self, x):
        f_x = self._sigmoid(x)
        return self.c * f_x * (1 - f_x)
    
    def _gaussian(self, x):
        return np.exp(-self.c * x**2)
    
    def _gaussian_grad(self, x):
        return -2 * self.c * x * self._gaussian(x)
    
    def _f(self, x):
        if self.eq_function == 'gaussian':
            return self._gaussian(x)
        else:
            return self._sigmoid(x)
    1
    def _grad_f(self, x):
        if self.eq_function == 'gaussian':
            return self._gaussian_grad(x)
        else:
            return self._sigmoid_grad(x)
    
    def basic_update_term(self, y_pred, y_true):
        score = abs(y_true - y_pred)
        
        err_t = 1 if score > self.q else 0

        diff = score - self.q
        
        return err_t - self.alpha + diff * self._grad_f(diff)

    def _compute_update_term(self, y_pred, y_true):
        if self.version == 'integral':
            return None
        elif self.version == 'cutoff':
            return None
        else:  # Basic version
            return self.basic_update_term(y_pred, y_true)
        
    
