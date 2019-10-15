#coding utf-8
import os
import numpy as np
from PIL import Image
import time
import svm
import csv
#图片矢量化


#读取数据
#
#@params:k 随机的参数
#

print('step 1 loading data ....')
olrpath='./data/orl_faces'
## 40人 每人选择其中的k个
def load_orl(k):
	train_face=np.zeros((40*k,112*92))
	train_label=np.zeros(40*k)
	test_face=np.zeros((40*(10-k),112*92))
	test_label=np.zeros(40*(10-k))
	
	#随机打乱位置
	#sample=np.random.permutation(10)+1

	for i in range(40):
		people_num = i+1
		for j in range(1,11):
			imgpath=olrpath+'/s'+str(people_num)+'/'+str(j)+'.pgm'
			img=Image.open(imgpath)
			data=np.mat(img.getdata())
			if j-1<k:
				#构成训练集
				train_face[i*k+j-1,:]=data
				train_label[i*k+j-1]=people_num
			else:
				#构成测试数据集
				test_face[i*(10-k)+(j-1-k),:]=data
				test_label[i*(10-k)+(j-1-k)]=people_num

	return train_face,train_label,test_face,test_label

# 定义PCA算法
#
# @param data train_data
# @param r n_components
#
def PCA(data,r):
	rows,cols=np.shape(data)
	data_mean=np.mean(data,0) #按列求均值
	A=data-np.tile(data_mean,(rows,1)) #中心化
	C=np.dot(A,A.T) #得到协方差矩阵 表征每两个元素之间的关联程度
	D,V=np.linalg.eig(C) #求协方差矩阵的特征值和特征向量 ？为何不对特征值进行排序
	V_r=V[:,0:r]#提取前r个最大特征值对应的特征向量
	V_r=np.dot(A.T,V_r)
	for i in range(r):
		V_r[:,i]=V_r[:,i]/np.linalg.norm(V_r[:,i])#特征向量归一化
	final_data=np.dot(A,V_r)
	return final_data,data_mean,V_r

# face_rec
# main function
def face_rec(k,n_components):
	print('降到40维的情况')
	train_face,train_label,test_face,test_label=load_orl(k)

	#使用PCA进行降维
	final_data,data_mean,V_r=PCA(train_face,n_components)
	#训练脸总数
	num_train = final_data.shape[0]
	#测试脸总数
	num_test = test_face.shape[0]
	#中心化测试脸
	temp_face = test_face - np.tile(data_mean,(num_test,1))
	#得到测试脸在特征向量下的数据
	data_test_new = np.dot(temp_face,V_r) 

	data_test_new = np.array(data_test_new) # mat change to array
	data_train_new = np.array(final_data)

	return data_train_new,train_label,data_test_new,test_label

	#测试精确度




train_x,train_y,test_x,test_y=face_rec(6,40)

## step 2 training ...
print(int(train_x.shape[0]/12))
for i in range(int(train_x.shape[0]/12)):
	print('')
	train_x=train_x[i*12:(i+12)*12,:]
	train_y=train_y[i*12:(i+12)*12]
#train_x=train_x[0:12,:]
#train_y=train_y[0:12]
test_x=test_x[0:8,:]
test_y=test_y[0:8]
print ("step 2: training..." ) 
C = 0.6  
toler = 0.001  
maxIter = 50  
svmClassifier = svm.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('linear', 0))  
#print(svmClassifier.alphas)
#print(svmClassifier.KernelMat)
#print(svmClassifier.b)
## step 2: training...  
'''
print ("step 2: training..." ) 
C = 0.6  
toler = 0.001  
maxIter = 50  
svmClassifier = svm.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('linear', 0))  
'''
print('step 3: testing')
accuracy=svm.testSVM(svmClassifier, test_x, test_y)
print ('The classify accuracy is: %.3f%%' % (accuracy * 100) ) 

### 推广 计算每两个数据之间的分类
### 
