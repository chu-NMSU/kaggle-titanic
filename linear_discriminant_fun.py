import numpy

def compute_tilde_X(X):
    tilde_X = numpy.insert(X, 0, 1, axis=1)
    return tilde_X

def compute_T(t, K):
    T = []
    for i in range(0, len(t)):
        T_i = K*[0]
        T_i[t[i]] = 1
        T.append(T_i)

    return numpy.array(T)#.transpose()

def LS_estimate(X, t, K):
    tilde_X = compute_tilde_X(X)
    T = compute_T(t, K)
    tilde_W = numpy.dot(numpy.dot(numpy.linalg.pinv(numpy.dot(tilde_X.transpose(), tilde_X)), tilde_X.transpose()), T)
    return tilde_W

def LS_predict(X, tilde_W):
    W_0 = tilde_W[:,0]
    W_1 = tilde_W[:,1]
    output = []
    for x_test in X:
        if numpy.dot(W_0, numpy.insert(x_test,0,1)) >= numpy.dot(W_1, numpy.insert(x_test,0,1)):
            output.append(0)
        else:
            output.append(1)
    return numpy.array(output)
     
def fisher_linear_discriminant(X, t):
    x_0, x_1 = X[t==0], X[t==1]
    mean0 = numpy.mean(x_0, axis=0)
    mean1 = numpy.mean(x_1, axis=0)
    mean = 0.5*(mean0+mean1)
    Sw = numpy.dot((x_0-mean0).T, (x_0-mean0)) + numpy.dot((x_1-mean1).T, (x_1-mean1)) 
    w = numpy.dot(numpy.linalg.pinv(Sw), (mean1-mean0))
    return w, mean

def fisher_linear_predict(X, w, mean, threshold=0):
    output = []
    for x_test in X:
        if numpy.dot(w, (x_test-mean)) > threshold:
            output.append(1)
        else:
            output.append(0)
    return numpy.array(output)
