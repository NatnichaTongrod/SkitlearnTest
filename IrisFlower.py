# #ทำการ import dataset
# from sklearn.datasets import load_iris

# irisFlower = load_iris()

# #แสดงข้อมูลของพวกขนาดกลีบเลี้ยง กลีบดอก
# print(irisFlower.feature_names)

# #แสดงข้อมูลสายพันธ์ของดอก iris
# print (irisFlower.target_names)

# #แสดง dataset แถวแรก
# print(irisFlower.data[0])

#*****************************************



import numpy as np
from  sklearn.datasets import load_iris
from sklearn import tree
irisFlower =load_iris()
 # test_idx เป็นตัวที่เก็บ id และ index 
 # ของสมาชิกตัวแรกในแต่ละสายพันธ์ของดอกไม้\
test_idx = [0,50,100]

#Training โดยการลบข้อมูล
#เป็นการทำลบค่าข้อมูลออกแล้วใส่กลับไปใหม่ดูว่าค่ายังเหมือนเดิมอยู๋ไหม
train_target = np.delete(irisFlower.target,test_idx)
train_data = np.delete(irisFlower.data, test_idx, axis = 0)

 #Testing โดยการเพิ่มข้อมูล
test_target = irisFlower.target[test_idx]
test_data = irisFlower.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target) #เพื่อ map target กับ data เข้าด้วยกัน


#output ตรงกันแสดงว่าข้อมูลใช้ได้
print(test_target) #output [0,1,2] นั่นคือประเภทของดอกไม้
print(clf.predict(test_data)) #output [0,1,2] 

import matplotlib.pyplot as plt
#เป็นการแสดง tree ข้างใน
tree.plot_tree(clf) 
plt.show()

