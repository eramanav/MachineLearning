# Load dataset
import pandas as pd
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
s=requests.get(url).text
dataset=pd.read_csv(StringIO(s))

#analyze the data
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())

dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
dataset.hist()
scatter_matrix(dataset)
pyplot.show()

#creating model
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()
