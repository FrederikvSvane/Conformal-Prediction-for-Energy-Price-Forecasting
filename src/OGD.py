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
    def __init__(self, alpha=0.1, eta=0.5, q_init=1.0, c=1.0, window_length=50, h=1, gamma=0.95, version='basic', 
                eq_function='sigmoid'):
        super().__init__(alpha, eta, q_init)
        self.c = c
        self.version = version
        self.eq_function = eq_function  # 'sigmoid' or 'gaussian'
        #Cutoff-specific parameters
        self.score_history = []
        self.window_length = window_length
        self.h = h
        # Integral-specific parameters
        self.error_history = []  # MISSING! Need to initialize this
        self.gamma = gamma  # Decay factor for weights

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
    
    def _grad_f(self, x):
        if self.eq_function == 'gaussian':
            return self._gaussian_grad(x)
        elif self.eq_function == 'sigmoid':
            return self._sigmoid_grad(x)
    
    def basic_update_term(self, y_pred, y_true):
        score = abs(y_true - y_pred)
        
        err_t = 1 if score > self.q else 0

        diff = score - self.q
        
        return err_t - self.alpha + diff * self._grad_f(diff)
    
    def cutoff_update_term(self, y_pred, y_true):
        score = abs(y_true - y_pred)
        
        # Store score in history
        self.score_history.append(score)
        
        # Keep only recent scores to save memory
        if len(self.score_history) > self.window_length * 2:
            self.score_history = self.score_history[-self.window_length:]
        
        err_t = 1 if score > self.q else 0
        
        if len(self.score_history) >= self.window_length:
            recent_scores = self.score_history[-self.window_length:]
        else:
            recent_scores = self.score_history
        
        # h_t = h * (max - min) of recent scores
        h_t = self.h * (max(recent_scores) - min(recent_scores))
        
        diff = score - self.q
        
        # Only apply EQ term if |diff| > h_t
        if abs(diff) > h_t:
            eq_term = diff * self._grad_f(diff)
        else:
            eq_term = 0
        
        return err_t - self.alpha + eq_term

    def integral_update_term(self, y_pred, y_true):
        """
        ECI-integral: Integrate errors over all past timesteps
        with exponentially increasing weights for recent errors
        """
        score = abs(y_true - y_pred)
        
        # Store current observation
        self.score_history.append(score)
        self.error_history.append({
            'score': score,
            'q': self.q,
            'err': 1 if score > self.q else 0
        })
        
        t = len(self.error_history)
        
        # Compute weights: w_i = 0.95^(t-i) / Σ 0.95^(t-j)
        gamma = 0.95  # Decay factor
        raw_weights = [gamma ** (t - i) for i in range(1, t + 1)]
        weight_sum = sum(raw_weights)
        weights = [w / weight_sum for w in raw_weights]
        
        # Compute weighted sum of all past update terms
        total_update = 0.0
        for i, (w_i, past) in enumerate(zip(weights, self.error_history)):
            err_i = past['err']
            s_i = past['score']
            q_i = past['q']
            diff_i = s_i - q_i
            
            # Update term for timestep i
            update_i = err_i - self.alpha + diff_i * self._grad_f(diff_i)
            
            # Weight it and add to total
            total_update += w_i * update_i
        
        return total_update

    def _compute_update_term(self, y_pred, y_true):
        if self.version == 'integral':
            return self.integral_update_term(y_pred, y_true)
        elif self.version == 'cutoff':
            return self.cutoff_update_term(y_pred, y_true)
        else:  # Basic version
            return self.basic_update_term(y_pred, y_true)
        
    
