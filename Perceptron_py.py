import numpy as np

class Perceptron(object):
    '''
    퍼셉트론 분류기

    1. 매개 변수 
        - eta(float) : 학습률(0.0 ~ 1.0)
        - n_iter(int) : 훈련 데이터셋 반복 횟수
        - random_state(int) : 가중치 무작위 초기화를 위한 난수 생성 시드
    2. 속성
        - w_(1d array) : 학습된 가중치
        - errors_(list) : 에포크마다 누적된 분류 오류
    '''

    def __init__ (self, eta=0.1, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        '''
        훈련 데이터 학습
        
        1. 매개변수
            - X : {array_like}, shape = [n_samples, n_features]
                n_samples개의 샘프과 n_features개의 특성으로 이루어진 훈련 데이터
            - y : array_like, shape = [n_samples]
                타깃 값

        2. 반환값
            - self : object 
        '''

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale=0.01, size = 1 + X.shape[1])

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0

            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            
            self.errors_.append(errors)

        return self

    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


