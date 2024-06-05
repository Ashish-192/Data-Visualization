import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the dataset
data = load_iris()
X = data.data
y = data.target

# Visualize the data using a scatter plot
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Iris dataset')
plt.show()

# Visualize the data using a histogram
plt.hist(X[:, 2], bins=50, color='green')
plt.xlabel('Petal length')
plt.ylabel('Frequency')
plt.title('Iris dataset')
plt.show()

# Visualize the data using a boxplot
plt.boxplot(X, labels=data.feature_names)
plt.xlabel('Features')
plt.ylabel('Value')
plt.title('Iris dataset')
plt.show()