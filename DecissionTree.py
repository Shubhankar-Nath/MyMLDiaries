from sklearn import tree
#[cc, seat, weight] as training data
X=[[1000, 6, 25000], [600, 4, 10000], [1200, 6, 20000], [700, 2, 9000], [1000, 6, 20000]]
Y=['SUV','Compact','SUV','Compact','SUV']
#Now initiating a classifier
clf=tree.DecisionTreeClassifier()
abc=clf.fit(X,Y)
#predicting output
prediction=abc.predict([[1000, 4, 25000]])
print prediction
