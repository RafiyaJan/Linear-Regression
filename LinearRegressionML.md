# Linear Regression

Linear regression is the supervised learning method that performs the Regression task when the target / dependent variable is continuous.Regression yields the desired prediction value premised on independent variables. It is mainly employed in determining the relationship between variables and prediction tasks. Different regression models differ in the type of relationship they compute between dependent and independent variables and the number of independent variables.
      Linear Regression is a predictive method that yields a linear relationship between the input (called X) and the prediction (Y^). It enables quantifying the relationship between a predictor and an output variable.
Machine learning systems employ linear regression to determine future values. It is the most common Machine Learning algorithm for making predictions and constructing models. Predictive analytics and modeling are the most common applications of linear regression in machine learning. It is premised on the principle of ordinary least square (OLS) / Mean square error (MSE). In statistics, OLS is employed to predict the unknown parameters of the linear regression function. Its objective is to minimize the sum of square differences between the observed dependent variables in the given data set and those predicted by linear regression function.

Its objective is to minimize the sum of square differences between the observed dependent variables in the given data set and those predicted by linear regression function. In order to predict  the continuous target variable we employ the following  notation:
  X’s to denote the input variables also called input features.
  Y’s to denote the output or target variable that we have to predict.
This implies, (X, Y) will denote one training example.
For some specific  ith  point, a pair (x<sup>(i)</sup>, y<sup>(i)</sup>) is called the training example and the dataset contains a list of n such training examples  where {x<sup>(i)</sup>, y<sup>(i)</sup>; i=1,2,….n)  } specifies training set. In a regression problem, we try to predict the target variable which is continuous and our objective is to learn a function h: X→y   (also called a hypothesis) such that h(x) is a good predictor for the corresponding value of y. 
Thus, mathematically  we can write 

##### Let **D={x<sup>(i)<sup>→</sup></sup>, y<sup>(i)</sup>)**  ∣ **x<sup>(i)</sup> ∈  X**,**y<sup>(i)</sup>∈ Y, i= 1,2,……N}**,<br>

#####  **X={ x<sup>→</sup> | x<sup>→</sup> ∈ ℝ<sup>d</sup> }** and **Y= { y  |  y ∈ ℝ}**
}

And x 's  are  the d-dimensional vectors and y is a real value to be predicted.The Linear regression model for d-dimensional data introduces a weight vector w =w<sub>1</sub>,w<sub>2</sub>,...,w<sub>d</sub> and bias value w0 to predict the output value as a linear combination of the input features x<sup>i</sup><sub>j</sub>  ( where x<sup>i</sup><sub>j</sub> denotes the j<sup>th</sup> feature of the i<sup>th</sup>  point )  of the input x<sup>i<sup>→</sup></sup>as
![image](https://user-images.githubusercontent.com/97376928/160909144-6d3e6a0f-1856-44ad-957d-8f123d6e2f17.png)<br>
Where w<sup>(k)</sup>’s are the parameters(also known as weights ) parameterizing the space of linear functions mapping from X to Y. Sometimes, we modify the  x<sup>→</sup> and  w<sup>→</sup>  vectors to introduce  x<sub>0</sub> = 1(intercept term )   and include  w<sub>0</sub>  into the weight vector  w<sup>→</sup>    such the above equation simplifies as: <br>
![image](https://user-images.githubusercontent.com/97376928/160911903-a26d727e-8c5e-4911-a9b2-b5081bcf441e.png)<br>
Here, the d is the number of input variables and limits of k change from 1 before modifying the vectors to 0 after modification.The elements of x<sup>→</sup>  are features of a data point and to find out how much a particular feature contributes towards the output i.e, their contribution is represented by the corresponding weight from   w<sup>→</sup> .
## Model Evaluation
To find out how well our model performs,we require a cost function. A good choice in this case is the squared error function (although the choice is not random and there is a reason behind choosing this function):
![image](https://user-images.githubusercontent.com/97376928/160912682-6027599a-7e32-427d-b14d-55eaa591253b.png)<br>
We can find the cost over all the dataset as:<br>
![image](https://user-images.githubusercontent.com/97376928/160913118-2bf7bd36-0013-41f0-85f5-b498f6bdf5d8.png)<br>
The scaling factor 1/2 before the summation  is to make the math easier as you will find out.
## Gradient Descent to find the minimum of a cost function
The cost function is<br>
![image](https://user-images.githubusercontent.com/97376928/160913118-2bf7bd36-0013-41f0-85f5-b498f6bdf5d8.png)<br>
and the goal is to minimize the J(w<sup>→</sup>,D).To do so we start with some initial guess for w,that repeatedly changes w to make J(w<sup>→</sup>,D) smaller,until we converge to a value of w that minimizes J(w<sup>→</sup>,D).
To achieve this we employ gradient descent to find the minimum of the cost function that starts with some initial value of w<sub>k</sub>. For a particular w<sub>k</sub> ,where  k=0,1,…,d , we have<br>
![image](https://user-images.githubusercontent.com/97376928/160915298-f62757ba-c3e0-451f-b2b2-5505b612aa89.png)<br>
Lets now first find <br>
![image](https://user-images.githubusercontent.com/97376928/160916773-67b7df9c-1abe-40f5-9583-c5877ab0281a.png)<br>and then substitute that in the above equation 1.<br>
![image](https://user-images.githubusercontent.com/97376928/160917253-892a1507-f548-4630-8387-6f302c938bee.png)<br>
Substituting this in the above equation 1   we get,<br>
![image](https://user-images.githubusercontent.com/97376928/160917533-962b5a2f-e35a-41c1-80b8-80a3ab6265f9.png)<br>
We will find the gradient with respect to all  w<sub>k</sub>'s  to form the gradient vector<br>
![image](https://user-images.githubusercontent.com/97376928/160917799-ef237415-f71d-4cd9-8c93-16ca970f5165.png)<br>
where  J=J(w<sup>→</sup>,D) for ease of notation.
Hence we have ,<br>
          <center>Loop until convergence</center><br>

![image](https://user-images.githubusercontent.com/97376928/160918871-393be701-dcda-4dd5-86ed-2c128a65ea78.png)<br>

## Vectorized Form
Matrices are computationally efficient for defining the linear regression model and performing the subsequent analyses.The above equation can be denoted in vectorized form which leads to ease in implementation.<br>Let us define the matrix  X  which contains all the input vectors  x<sup>→</sup>  along its rows as:<br>
![image](https://user-images.githubusercontent.com/97376928/160919232-0cc96b44-c96b-4fd7-b7c1-827090e94d10.png)<br>
For two vectors  x<sup>→</sup> , y<sup>→</sup>  ∈ ℝ<sup>d</sup>  , we have<br>
![image](https://user-images.githubusercontent.com/97376928/160919539-d01851f8-ebe2-477d-b917-158e7f16eee8.png)<br>
 Where x<sub>k</sub><sup>→</sup> is the k-th column X matrix. That is, the k-th entry from each input vector x<sub>k</sub><sup>→</sup>.<br>

![image](https://user-images.githubusercontent.com/97376928/160920278-834219cb-4d9c-4c83-b953-fd4f1f68c00a.png)<br>To find the 
![image](https://user-images.githubusercontent.com/97376928/160920567-98990b27-67a6-4731-9627-d128cdbdad6c.png)<br>
we have:
![image](https://user-images.githubusercontent.com/97376928/160920700-3c79a84a-0134-4c4f-8df2-36700ae70d89.png)

## Normal equations 
 
  We have derived above the expression for <br>
  ![image](https://user-images.githubusercontent.com/97376928/160920936-a673db35-6f2e-489c-8d5c-648b6600e657.png)
<br>
  Now to find the minimum using normal equations, we have<br>
  ![image](https://user-images.githubusercontent.com/97376928/160921206-36d5d2ec-211a-4c08-891c-c52602196cbd.png)<br>
  ## Implementation of Linear Regression Using Python code
### Dataset
We employ Diabetes dataset that has 442 data points and each data point has 10 attributes. The target is a quantitative measure of disease progression one year after baseline.More details can be found [here](https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf). Scikit-learn has this dataset and we will get the dataset from there.
### Python Code
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
dia_data = load_diabetes()
print(dir(dia_data))
X_dia = dia_data.data
Y_dia = dia_data.target
print(f"Number of samples is:{len(X_dia)}")
print(f"Number of attributes is:{X_dia[0].shape}")
print(f"The attributes are {dia_data.feature_names}")
print(f"The min and max of target is {np.min(Y_dia), np.max(Y_dia)}")
````
#### Output
>['DESCR', 'data', 'data_filename', 'data_module', 'feature_names', 'frame', 'target', 'target_filename']<br>
Number of samples is:442<br>
Number of attributes is:(10,)<br>
The attributes are ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']<br>
The min and max of target is (25.0, 346.0)<br>
We will not work with all the attributes rather we will use only the bmi and bpto predict the output. The output is a continuous value.

```python
X_dia = X_dia[:,2:4]<br>
print(f"X_dia shape is {X_dia.shape}")<br>
print(f"Min and max of X_dia is {np.min(X_dia,0), X_dia.max(0)}")
````
#### Output
>X_dia shape is (442, 2)<br>
Min and max of X_dia is (array([-0.0902753, -0.1123996]), array([0.17055523, 0.13204422]))

```python
from mpl_toolkits import mplot3d
fig = plt.figure
ax = plt.axes(projection="3d")
ax.scatter(X_dia[:,0], X_dia[:,1], Y_dia)
````
#### Output
>![image](https://user-images.githubusercontent.com/97376928/160990272-2c90ab61-d87c-4454-9e7e-009211a3338c.png)

```python
from sklearn.model_selection import train_test_split
X_dia_train, X_dia_test, Y_dia_train, Y_dia_test = train_test_split(X_dia, Y_dia, train_size=.8)
````
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_dia_train[:,0].reshape([-1,1]), Y_dia_train)
````
```python
Y_dia_pred = model.predict(X_dia_test[:,0].reshape([-1,1]))
````
```python
from sklearn.metrics import mean_squared_error as mse
error = mse(Y_dia_test, Y_dia_pred)
````
#### Output
>3520.476715737209
### Python Code
```python
plt.plot(X_dia_test, Y_dia_test, 'ko')
plt.xlabel("bmi",fontname="serif", fontsize=16)
plt.ylabel("Y",fontname="serif", fontsize=16)
plt.title("Logistic Regression on Diabetes Dataset",fontsize=18, fontname="serif",style='italic')
````
#### Output
>![image](https://user-images.githubusercontent.com/97376928/160990626-d6e36ff9-c277-40b2-9cd6-211aa45d8868.png)

