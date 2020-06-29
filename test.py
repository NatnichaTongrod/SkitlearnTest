from sklearn import tree
features =[[120,5],[98,5],[180,12],[163,12]]
labels = ["ECO CAR / B-SEGMENT","ECO CAR / B-SEGMENT","VAN","VAN"]
classifier  = tree.DecisionTreeClassifier()

# fit เป็นการหา pattern ของข้อมูลที่เราส่งเข้าไป
classifier = classifier.fit(features,labels)
print(classifier.predict([[98,5]])) 
