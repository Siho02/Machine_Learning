import numpy as np

class AdalineGD(object):
    '''
    1. 매개변수
        - eta : flaot
            학습률(0.0 ~ 1.0)
        - n_iter : int
            훈련 데이터셋 반복 횟수
        - random_state : int
            가중치 무작위 초기화를 위한 난수 생성 시드
    2. 속성
        - w_ : 1d-array
            학습된 가중치
        - cost_ : list
            에포크마다 누적된 비용함수의 제곱합
    '''

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return X
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
    
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size = 1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        
        return self 
