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


