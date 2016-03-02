__author__ = 'Administrator'

# name = raw_input('Your name:')
# print name, ', hello world!'

from sklearn.datasets import load_iris

data = load_iris()
features = data['data']
target = data['target']

print '=========================='
print type(data)
print data
#print features
#print target

