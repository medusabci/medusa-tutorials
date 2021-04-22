import numpy as np
from medusa import components as cmp
from medusa.timer import timer

# Define some functions
def suma(a, b, c):
    return np.sum([a, b, c])


class Resta(cmp.ProcessingMethod):

    def __init__(self):
        super().__init__(fit=[], apply=['res'])
        self.r = None

    def fit(self, r):
        self.r = r

    def apply(self, n):
        return n - self.r

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(dict_data):
        pass


# Methods
suma_wrap = cmp.ProcessingMethodFuncWrapper(func=suma, outputs=['res'])
resta_r = Resta()

# Train pipe
train_pipe = cmp.Pipeline()
uid_0 = train_pipe.input(['a', 'b', 'c'])
uid_1 = train_pipe.add(method_func_key='suma',
                       a=train_pipe.conn_to(uid_0, 'a'),
                       b=train_pipe.conn_to(uid_0, 'b'),
                       c=train_pipe.conn_to(uid_0, 'c'))
uid_2 = train_pipe.add(method_func_key='suma',
                       a=train_pipe.conn_to(uid_0, 'a'),
                       b=train_pipe.conn_to(uid_0, 'b'),
                       c=train_pipe.conn_to(uid_1, 'res'))
uid_3 = train_pipe.add(method_func_key='resta:fit',
                       r=train_pipe.conn_to(uid_2, 'res'))

# Test pipe
test_pipe = cmp.Pipeline()
uid_0 = test_pipe.input(['a', 'b', 'c'])
uid_1 = test_pipe.add(method_func_key='suma',
                      a=test_pipe.conn_to(uid_0, 'a'),
                      b=test_pipe.conn_to(uid_0, 'b'),
                      c=test_pipe.conn_to(uid_0, 'c'))
uid_2 = test_pipe.add(method_func_key='resta:apply',
                      n=test_pipe.conn_to(uid_1, 'res'))

# Algorithm
alg = cmp.Algorithm()

# Add methods
alg.add_method('suma', suma_wrap)
alg.add_method('resta', resta_r)

# Add pipeline
alg.add_pipeline('train', train_pipe)
alg.add_pipeline('test', test_pipe)

# Execute pipeline
train_res = alg.exec_pipeline('train', a=1, b=1, c=1)
test_res = alg.exec_pipeline('test', a=2, b=1, c=1)

# Test func wrappers
cmp.ProcessingMethodFuncWrapper.from_dict(suma_wrap.to_dict())

# Test alg
d = alg.to_dict()
