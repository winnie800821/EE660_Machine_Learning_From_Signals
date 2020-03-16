import numpy as np
import matplotlib.pyplot as plt


Xtrain = np.loadtxt(open("x_train.csv"), delimiter=",")
Ytrain = np.loadtxt(open("y_train.csv"), delimiter=",")
Xtrain=np.array([Xtrain])

Xtest = np.loadtxt(open("x_test.csv"), delimiter=",")
Ytest = np.loadtxt(open("y_test.csv"), delimiter=",")
Xtest=np.array([Xtest])
plt.scatter(Xtrain,Ytrain)
plt.title("x_train vs. y_train Plot")
plt.xlabel('x_train')
plt.ylabel('y_train')
plt.show()

#Question1.(d)~(f)
print('Question1.(d)~(f)')
degree=[1,2,3,7,10]
MSE_array_train=[]
MSE_array_test=[]
for i in degree:
    print(' ')
    print('When degree=',i)
    arrayX=np.array([np.ones(Xtrain.shape[1])])
    arrayX_test=np.array([np.ones(Xtest.shape[1])])
    t=1
    while t<=i:
        a = np.power(Xtrain, t)
        b = np.power(Xtest,t)
        arrayX = np.append(arrayX, a, axis=0)
        arrayX_test = np.append(arrayX_test, b, axis=0)
        t+=1
    arrayX=np.transpose(arrayX)
    arrayX_test=np.transpose(arrayX_test)
    XTX=(np.transpose(arrayX)).dot(arrayX)
#    print('arrayX=',arrayX)
#    print(arrayX.shape)
    pseu_X=(np.linalg.inv(XTX)).dot(np.transpose(arrayX))

    w = pseu_X.dot(Ytrain)
    print('w',i,'=',w)
    diff_train=(Ytrain-arrayX.dot(w))**2
    diff_test=(Ytest-arrayX_test.dot(w))**2
    #compute the MSE for train data
    RSS_train=0
    for k in diff_train:
        RSS_train=RSS_train+k
    MSE_train=RSS_train/Xtrain.shape[1]
    MSE_array_train.append(MSE_train)
    print('MSE(train)=', MSE_train)
    # compute the MSE for test data
    RSS_test=0
    for k in diff_test:
        RSS_test=RSS_test+k
    MSE_test=RSS_test/Xtest.shape[1]
    MSE_array_test.append(MSE_test)
    print('MSE(test)=',MSE_test)

plt.scatter(degree,MSE_array_train,'o-',color = 'b',label="MSE for the training set")
plt.scatter(degree,MSE_array_test,'s-',color = 'r',label="MSE for the test samples")
plt.title("error vs. polynomial degree Plot")
plt.xlabel('polynomial degree')
plt.ylabel('error')
plt.legend(loc = "best")
plt.show()
print(' ')
#Question1.(g)~(h)
print('Question1.(g)~(h)')
#w=(lambda*I+XT*X)^(-1)*XT*y
arrayX=np.array([np.ones(Xtrain.shape[1])])
arrayX_test=np.array([np.ones(Xtest.shape[1])])
t=1
while t<=7:
    a = np.power(Xtrain,t)
    b = np.power(Xtest,t)
    arrayX = np.append(arrayX, a, axis=0)
    arrayX_test = np.append(arrayX_test, b, axis=0)
    t+=1
arrayX=np.transpose(arrayX)
arrayX_test=np.transpose(arrayX_test)

XTX=np.transpose(arrayX).dot(arrayX)
I=np.identity(XTX.shape[0])
lamb=[1/(10**5), 1/(10**3), 1/10, 1, 10]
MSE_ridge_train=[]
MSE_ridge_test=[]
for i in lamb:
    pseu_X=(np.linalg.inv(i*I+XTX)).dot(np.transpose(arrayX))
    w_ridge=pseu_X.dot(Ytrain)
    print(' ')
    print('When λ=',i)
    print('w=',w_ridge)
    diff_train = (Ytrain - arrayX.dot(w_ridge)) ** 2
    diff_test = (Ytest - arrayX_test.dot(w_ridge)) ** 2
# compute the MSE for train data
    MSE_train = sum(diff_train) / diff_train.size
    MSE_ridge_train.append(MSE_train)
    print('MSE(train)=', MSE_train)
# compute the MSE for test data
    MSE_test = sum(diff_test) / diff_test.size
    MSE_ridge_test.append(MSE_test)
    print('MSE(test)=', MSE_test)


plt.plot(np.log10(lamb),MSE_ridge_train,'o-',color = 'b',label="MSE for the training set")
plt.plot(np.log10(lamb),MSE_ridge_test,'s-',color = 'r',label="MSE for the test samples")
plt.title("MSE vs. log(λ) Plot  (Ridge Regression)")
plt.xlabel('log(λ)')
plt.ylabel('MSE')
plt.legend(loc = "best")
plt.show()




