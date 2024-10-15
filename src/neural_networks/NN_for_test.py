import numpy as np
import matplotlib.pyplot as plt
import sklearn  
import sklearn.datasets
import sklearn.linear_model
import matplotlib

# Display plots inline and change default figure size 
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

np.random.seed(3)
X, y = sklearn.datasets.make_moons(200, noise=0.2)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.title("half_moon")
plt.show()

# adopt logistic regression to carry on the classification
lrclf = sklearn.linear_model.LogisticRegressionCV()
lrclf.fit(X, y)

# helper function to draw a decision boundary
def plot_decision_boundary(pred_func): 
    # Set min and max values and give it some padding 
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5 
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5 
    h = 0.01 
    # Generate a grid of points with distance h between them 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
    # Predict the function value for the whole gid 
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape) 
    plt.figure()
    # Plot the contour and training examples 
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral) 
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    
plot_decision_boundary(lambda x: lrclf.predict(x))
plt.title("Logistic Regression")
plt.show()

# parameters initialization
def initialize_parameters(input_dim, hidden_dim, output_dim):
    W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(2 / (input_dim + hidden_dim))
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(2 / (hidden_dim + output_dim))
    b2 = np.zeros((1, output_dim))
    
    return W1, b1, W2, b2

# forward propagation
def forward_propagation(X, W1, b1, W2, b2):
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    return probs, a1

# back propagation
def back_propagation(X, y, probs, a1, W2):
    delta3 = probs
    delta3[range(X.shape[0]), y] -= 1
    dW2 = (a1.T).dot(delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)
    delta2 =  delta3.dot(W2.T) * (1 - np.power(a1, 2))
    dW1 = (X.T).dot(delta2)
    db1 = np.sum(delta2, axis=0)

    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    new_model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    
    return new_model

def get_accuracy(model, X, y):
    y_pred = predict(model, X)
    
    return (y_pred == y).sum() / X.shape[0]

def predict(model, X):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    probs, _ = forward_propagation(X, W1, b1, W2, b2)
    
    return np.argmax(probs, axis=1)

# model training and result producing
def bpnn_clf(X, y, input_dim, hidden_dim, output_dim, learning_rate):
    W1, b1, W2, b2 = initialize_parameters(input_dim, hidden_dim, output_dim)
    i = 0
    model = {}
    best_accuracy = 0.995
    
    while True:
        i += 1
        probs, a1 = forward_propagation(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = back_propagation(X, y, probs, a1, W2)
        model = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        accuracy = get_accuracy(model, X, y)
        print(f"{i} iteration, accuracy: {accuracy}")
        
        if accuracy < best_accuracy:
            if i > 50000:
                print(f"Can't achieve best accuracy")
                break
            continue
        else:
            print(f"after {i} iterations, best accuracy:{accuracy},\nbest model: {model}")
            plot_decision_boundary(lambda X: predict(model, X))
            plt.show()
            break
        

input_dim = X.shape[1]
hidden_dim = 20
output_dim = 2
bpnn_clf(X, y, input_dim, hidden_dim, output_dim, 0.001)
