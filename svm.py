import numpy as np
import time
import matplotlib.pyplot as plt


# 定义SVM 结构类型
# @param dataSet
# @param labels
# @param C
# @param toler
# @param kernelOption
#


class SVMstruct:
	def __init__(self,dataSet,labels,C,toler,KernelOption):
		self.train_x=dataSet
		self.train_y=labels
		self.C=C
		self.numSameples=dataSet.shape[0]
		self.alphas=np.mat(np.zeros((self.numSameples,1))) #所有的拉格朗日乘数 初始化为全0的矩阵
		self.toler=toler #迭代的终止条件 容错率
		self.b=0
		self.errorCache=np.mat(np.zeros((self.numSameples,2)))
		self.KernelOption=KernelOption
		self.KernelMat=calcKernelMatrix(self.train_x, self.KernelOption)  

#核心 SMO
# 利用SMO 求解 alpha


# calaculate kernel value
# 计算核值
#
# @param matrix_x 数据集
# @param sample_x 样本集
# @param kernelOption 核函数 选项
#

def calcKernelValue(matrix_x,sample_x,KernelOption):
	kernelType = KernelOption[0]  
	numSamples = matrix_x.shape[0]  
	kernelValue = np.mat(np.zeros((numSamples, 1)))  

	if kernelType=='linear':
		kernelValue = np.dot(matrix_x,sample_x.T)
	elif kernelType=='rbf':
		sigma = KernelOption[1]  
		if sigma == 0:  
			sigma = 1.0  
		for i in range(numSamples):  
			diff = matrix_x[i, :] - sample_x  
			kernelValue[i] = exp(diff * diff.T / (-2.0 * sigma**2))  
	else:  
		raise NameError('Not support kernel type! You can use linear or rbf!')  
	return kernelValue  


# calculate kernel matrix given train set 
# 计算核矩阵-- 方便查询
# 核矩阵存在定理 --- 只要一个对称函数的核矩阵是半正定的则这个函数可以作为核函数使用
#
# @param train_x 
# @param kernelOption 核函数选项
# @return kernelMatrix
#  

def calcKernelMatrix(train_x,kernelOption):
	numSameples=train_x.shape[0]
	KernelMatrix=np.mat(np.zeros((numSameples,numSameples)))
	for i in range(numSameples):
		KernelMatrix[:,i]=calcKernelValue(train_x,train_x[i,:],kernelOption)
	return KernelMatrix

# calculate the error for aloha_k
# error Ei=f(xi)-yi

def calcError(svm,alpha_k):
	output_k = float(np.multiply(svm.alphas,svm.train_y.T).T * svm.KernelMat[:, alpha_k] + svm.b)  
	error_k=output_k-float(svm.train_y.T[alpha_k])	
	return error_k

# update the error temp for alpha k after optimize alpha k
### ??? 目的是
### 仅仅只是为了记录整个Error
def updateError(svm,alpha_k):
	error=calcError(svm,alpha_k)
	svm.errorCache[alpha_k]=[1,error]


# select alpha j which has the  biggest interval
### ??? 选择依据不是选择违背KTT条件最大的
### answer: alpha_i 选取KTT 违背程度最大 alpha_j 选取使得目标函数增长最快的
### 但出于算法复杂度的考虑 选取两个变量对应样本之间的距离间隔为最大 
### 这样两个样本的差异比较大 对这两个进行更新有助于带给目标函数更多的变化
### 计算变量对应样本的误差值是一个道理

# 寻找间隔最大的两个样本
def selectAlpha_j(svm,alpha_i,error_i):
	svm.errorCache[alpha_i]=[1,error_i] # mark as valid(has been optimized)  
	# 寻找有误差的 alpha 
	candidateAlphaList=np.nonzero(svm.errorCache[:,0].A)[0] #retrun mat.A array 返回标志位非零的位置 
	# 初始化参数
	maxInterval=0;
	alpha_j=0;
	error_j=0

	# find the alpha with the max iterative step
	if len(candidateAlphaList)>1:
		for alpha_k in candidateAlphaList:
			if alpha_k==alpha_i:
				continue
			error_k=calcError(svm,alpha_k)
			if abs(error_k-error_i)>maxInterval:
				maxInterval=abs(error_k-error_i)
				alpha_j=alpha_k
				error_j=error_k
	  # if came in this loop first time, we select alpha j randomly 
	else:
		alpha_j=alpha_i
		while alpha_j==alpha_i:
			alpha_j=int(np.random.uniform(0,svm.numSameples))
		error_j=calcError(svm,alpha_j)

	return alpha_j,error_j




# inner loop for optimizing alpha_i and alpha_j
def innerLoop(svm,alpha_i):
	# step1 calc error rate
	error_i = calcError(svm,alpha_i)
	### check and pick up the alpha who violates KTT condition
	### first of all
	### satify KTT condition
	# 1) yi*f(i) >= 1 and alpha == 0 (outside the boundary)  
    # 2) yi*f(i) == 1 and 0<alpha< C (on the boundary)  
    # 3) yi*f(i) <= 1 and alpha == C (between the boundary)  
     ## violate KKT condition  
    # because y[i]*E_i = y[i]*f(i) - y[i]^2 = y[i]*f(i) - 1, so  
    # 1) if y[i]*E_i < 0, so yi*f(i) < 1, if alpha < C, violate!(alpha = C will be correct)   
    # 2) if y[i]*E_i > 0, so yi*f(i) > 1, if alpha > 0, violate!(alpha = 0 will be correct)  
    # 3) if y[i]*E_i = 0, so yi*f(i) = 1, it is on the boundary, needless optimized  
	if(svm.train_y.T[alpha_i]*error_i < -svm.toler) and (svm.alphas[alpha_i] < svm.C ) or (svm.train_y.T[alpha_i]*error_i>svm.toler) and (svm.alphas[alpha_i] >0):
    	#step 1 select alpha j
		alpha_j,error_j=selectAlpha_j(svm,alpha_i,error_i)
    	### ??? 为什莫要做浅拷贝
		alpha_i_old = svm.alphas[alpha_i].copy()  
		alpha_j_old = svm.alphas[alpha_j].copy()  

		#step2:calculate the boundary l and H for alpha j
		if svm.train_y.T[alpha_i]!=svm.train_y.T[alpha_j]:
			L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])  
			H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])  
		else:
			L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)  
			H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])  
		if L==H:
			return 0

		#step3 :calculate eta (the similartiy of sample i and j)
		## eta 学习速率 i 和 j 的相似度
		eta = 2.0 * svm.KernelMat[alpha_i, alpha_j] - svm.KernelMat[alpha_i, alpha_i] - svm.KernelMat[alpha_j, alpha_j]  
		if eta >= 0:  
			return 0  

		# step4 update alpha j
		svm.alphas[alpha_j] -= svm.train_y.T[alpha_j] * (error_i - error_j) / eta  

		#step5 clip alpha j
		if svm.alphas[alpha_j] > H:  
			svm.alphas[alpha_j] = H  
		if svm.alphas[alpha_j] < L:  
			svm.alphas[alpha_j] = L  

		# step 6: if alpha j not moving enough, just return       
		if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:  
			updateError(svm, alpha_j)  
			return 0  

		# step 7: update alpha i after optimizing alpaha j
		svm.alphas[alpha_i] += svm.train_y.T[alpha_i] * svm.train_y.T[alpha_j] * (alpha_j_old - svm.alphas[alpha_j])  

		# step 8: update threshold b
		b1 = svm.b - error_i - svm.train_y.T[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old)* svm.KernelMat[alpha_i, alpha_i] \
		*svm.train_y.T[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
		* svm.KernelMat[alpha_i, alpha_j]  
		b2 = svm.b - error_j - svm.train_y.T[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
		* svm.KernelMat[alpha_i, alpha_j] \
		- svm.train_y.T[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
		* svm.KernelMat[alpha_j, alpha_j]  
		if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C):  
			svm.b = b1  
		elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C):  
			svm.b = b2  
		else:  
			svm.b = (b1 + b2) / 2.0  
  		# step 9: update error cache for alpha i, j after optimize alpha i, j and b  
		updateError(svm,alpha_i)
		updateError(svm,alpha_j)

		return 1
	else:
		return 0




# main function 
# 主函数 训练函数 
# @param train_x 数据集
# @param train_y 类别标签
# @param C 常数? -->C*epsilon = 松弛变量
# @param toler 容错率
# @param maxIter：退出前的循环次数 

def trainSVM(train_x,train_y,C,toler,maxIter,kernelOption=('rbf',1.0)):
	# start time 
	startTime=time.time()

	#svm struce
	svm=SVMstruct(np.mat(train_x), np.mat(train_y), C, toler, kernelOption)

	#start training
	entrieSet = True
	alphaPairsChanged=0
	iterCount=0

	# Iterator termination  condition:
	# 	condition1: reach  max iteration
	#   condition2: no alpha changed 
	#			or: all alpha fir KTT
	while(iterCount<maxIter) and ((alphaPairsChanged>0) or entrieSet):
		alphaPairsChanged=0

		#update alphas over all training example
		if entrieSet:
			for i in range(svm.numSameples):
				alphaPairsChanged+= innerLoop(svm, i) 
				print ('---iter:%d entire set, alpha pairs changed:%d' % (iterCount, alphaPairsChanged)  )
			iterCount += 1   
		# update alphas over examples where alpha is not 0 & not C (not on boundary)  
		else:
			nonBoundAlphasList=np.nonzero((svm.alphas.A > 0) * ( svm.alphas.A < svm.C))[0]
			for i in nonBoundAlphasList:
				alphaPairsChanged+=innerLoop(svm,i)
			print ('---iter:%d non boundary, alpha pairs changed:%d' % (iterCount, alphaPairsChanged)  )
			iterCount += 1

		 # alternate loop over all examples and non-boundary examples  
		if entrieSet:  
			entrieSet = False  
		elif alphaPairsChanged == 0:  
			entrieSet = True  
  
	print ('Congratulations, training complete! Took %fs!' % (time.time() - startTime))  
	return svm  

#testing your trained svm model given test set
def testSVM(svm,test_x,test_y):
	test_x=np.mat(test_x)
	test_y=np.mat(test_y)
	numTestSamples=test_x.shape[0]
	###?? 如何寻找的支持向量索引
	supportVectorsIndex = np.nonzero(svm.alphas.A > 0)[0]  
	supportVectors      = svm.train_x[supportVectorsIndex]  
	supportVectorLabels = svm.train_y.T[supportVectorsIndex]  
	supportVectorAlphas = svm.alphas[supportVectorsIndex]  
	matchCount = 0  
	for i in range(numTestSamples):  
		kernelValue = calcKernelValue(supportVectors, test_x[i, :], svm.KernelOption)  
		predict = kernelValue.T * np.multiply(supportVectorLabels, supportVectorAlphas) + svm.b  
		print('predict:',predict)
		if np.sign(predict) == np.sign(test_y.T[i]):  
			matchCount += 1  
	print(matchCount)
	accuracy = float(matchCount) / numTestSamples  
	return accuracy  

  
			