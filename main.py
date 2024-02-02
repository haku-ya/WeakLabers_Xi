import numpy as np 
import argparse
import sys
import pickle
import torch
import random
import itertools

import torch.utils.data as data
from torchvision.models import resnet

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# import data
import domain_net.wl as DWL
import domain_net.mini_domainnet as MDN
import aa2.aa2_data as AA2

# import algorithms
import amcl.algorithms.subgradient_method as SG
import amcl.algorithms.max_likelihood as ML

# baseline implementations
import labelmodels.labelmodels.naive_bayes as NB
import labelmodels.labelmodels.semi_supervised as SS
import all_algo as ALL
import heuristic_algo as PGMV

np.set_printoptions(threshold=sys.maxsize)

def str2bool(v):
	'''
	Used to help argparse library 
	'''
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def resnet_transform(unlabeled_data):
	'''
	Function to transform unlabeled data into features learned by 
	pre-trained resnet

	Args:
	unlabeled_data - raw pixel data
	'''

	ul1 = unlabeled_data[:300]
	ul2 = unlabeled_data[300:]
	res = resnet.resnet18(pretrained=True)
	td1 = res(torch.tensor(ul1))
	td1 = td1.detach().numpy()

	td2 = res(torch.tensor(ul2))
	td2 = td2.detach().numpy()

	transformed_data = np.concatenate([td1, td2])
	return transformed_data

def compute_avg_briar(y, y_pred, C):
	'''
	Function to compute the average briar loss over each example

	Args:
	y - true labels (one-hots)
	y_pred - prediction from the model (probability distribution)
	C - number of classes
	'''

	vals = []
	one_hots = np.eye(C)[y]

	# print(np.shape(y_pred), np.shape(one_hots))

	for i in range(len(y)):
		vals.append(SG.Brier_loss_linear(one_hots[i], y_pred[i]))

	return np.mean(vals)

def eval_comb(votes, labels, theta):
	'''
	Function to compute the accuracy of a weighted combination of labelers
	
	Args:
	votes - weak supervision source outputs
	labels - one hot labels
	theta - the weighting given to each weak supervision source
	'''

	N, M, C = np.shape(votes)
	totals = np.zeros((M, C))
	for i, val in enumerate(theta):
		for j, vote in enumerate(votes[i]):
			totals[j] += val * vote

	preds = np.argmax(totals, axis=1)
	briar_loss = compute_avg_briar(labels, totals, C)
	# print(np.mean(preds == true_labels))
	# print(confusion_matrix(true_labels, preds, labels=list(range(10))))
	return np.mean(preds == labels), briar_loss

def eval_lr(data, labels, theta, C):
	'''
	Function to evaluate a logistic regression model 

	Args:
	data - the data to evaluate the logreg model on
	labels - one hot labels
	theta - the weights for the logreg model
	C - the number of target classes
	'''

	probs = []
	preds = []
	for i, d in enumerate(data):
		p = SG.logistic_regression(theta, d)
		preds.append(np.argmax(p))
		probs.append(p)
	
	probs = np.array(probs)
	preds = np.array(preds)

	briar_loss = compute_avg_briar(labels, probs, C)
	return np.mean(preds == labels), briar_loss

def eval_sub(votes, sub):
	'''
	Function to evaluate the majority vote of a subset for binary tasks
	
	Args:
	votes - weak supervision source outputs
	sub - the subset of weak supervision sources to contribute to the majority vote
	'''

	N, M, C = np.shape(votes)
	probs = np.zeros((M, C))

	for i in sub:
		probs += votes[i]
	
	return probs / len(sub), np.argmax(probs, axis=1)


def compute_errors(votes, labels):
	'''
	Function to compute errors from votes of dimension (N, M, C)
	
	Args:
	votes - weak supervision source outputs
	labels - ground truth labels
	'''
	N, M, C = np.shape(votes)
	errors = np.zeros(N)
	for i in range(N):
		preds = np.argmax(votes[i], axis=1)
		errors[i] = np.mean(preds == labels)

	return errors

def correct_votes(errors, votes, vote_signals):
	'''
	Funciton to correct the votes for weak supervision sources with less than 50% accuracy
	(only in binary classificaiton tasks AwA2)

	Args:
	errors - error rates
	votes - weak supervision source outputs
	vote_signals - weak superivsion source outputs (soft classifications)
	'''
	for i, e in enumerate(errors):
		if e < 0.5:
			errors[i] = 1 - e
			votes[:, i] = np.where(votes[:, i] == 0, 1, 0)
			
			# swapping signal values
			M, C = np.shape(vote_signals[i])
			new_sigs = np.zeros((M, 2))
			new_sigs[:, 0] = vote_signals[i][:, 1]
			new_sigs[:, 1] = vote_signals[i][:, 0]
			vote_signals[i] = new_sigs
	return errors, votes, vote_signals

def convert_loaders(trainloader, testloader, unlab=500):
	'''
	Convert dataloaders into two new loaders, where the test loader has a specific
	number of examples

	Args:
	trainloader - training data loader
	testloader - testing data loader
	unlab - number of test data for new loader
	'''

	X = []
	Y = []
	
	for batch_x, batch_y in trainloader:
		for x in batch_x:
			X.append(x.numpy())
		for y in batch_y:
			Y.append(y.numpy())
	
	for batch_x, batch_y in testloader:
		for x in batch_x:
			X.append(x.numpy())
		for y in batch_y:
			Y.append(y.numpy())
	
	X = np.array(X)
	Y = np.array(Y)

	ratio = unlab / len(X)

	ss = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=0)
	for lab_index, unlab_index in ss.split(X, Y):

		print(len(lab_index), len(unlab_index))

		unlab_X, unlab_Y = X[unlab_index], Y[unlab_index]
		lab_X, lab_Y = X[lab_index], Y[lab_index]

		lab_loader = data.DataLoader([(torch.tensor(lab_X[i]), torch.tensor(lab_Y[i])) for i in range(len(lab_X))], shuffle=False, batch_size=50)
		unlab_loader = data.DataLoader([(torch.tensor(unlab_X[i]), torch.tensor(unlab_Y[i])) for i in range(len(unlab_X))], shuffle=False, batch_size=50)
		return lab_loader, unlab_loader

def aa2_experiment(task_ind, lab=100, sg=False, baseline=False, ind=False, log_reg=False, pgmv=False, all_algo=False):
	'''
	Function to run experiments on the AwA2 dataset

	Args:
	task_ind - index of a binary AwA2 task 
	'''

	# hardest examples by avg majority vote accuracy less than 80%
	hard_experiments = [[5, 0.77], [6, 0.63], [11, 0.527], [12, 0.38] , [15, 0.725], [20, 0.508], [24, 0.68], [25, 0.516], [28, 0.321], [41, 0.366]] 
	sorted_experiments = sorted(hard_experiments, key=lambda x: x[1])
	print(sorted_experiments)

	comb_ind = sorted_experiments[task_ind - 1][0]

	# getting task information defined by start
	unseen_classes = AA2.get_test_classes()
	combs = list(itertools.combinations(range(10), 2))
	classes = combs[comb_ind - 1]
	unseen = [unseen_classes[classes[0]], unseen_classes[classes[1]]]
	task = str(classes[0]) + str(classes[1])
	print("Task: " + task)

	features = AA2.get_feature_diffs(unseen)
	(labeled_X, labeled_labels, train_names), (unlabeled_X, unlabeled_labels, test_names) = AA2.gen_unseen_data_split(classes, 0)

	# hard labelers
	# train_votes = AA2.get_votes(classes, features, train_names)
	# test_votes = AA2.get_votes(classes, features, test_names)

	# soft labelers
	labeled_votes = AA2.get_signals(classes, features, train_names)
	unlabeled_votes = AA2.get_signals(classes, features, test_names)

	N = len(features) # from five domains
	C = 2 # 5 classes in test sample

	# Unlabeled data and Labeled data
	num_lab = np.shape(labeled_votes)[1]
	num_unlab = np.shape(unlabeled_votes)[1]

	# use fraction of labeled data
	lab_indices = random.sample(list(range(num_lab)), lab)
	
	labeled_X = labeled_X[lab_indices]
	labeled_votes = labeled_votes[:, lab_indices]
	labeled_labels = labeled_labels[lab_indices]

	labeled_labels = np.eye(2)[labeled_labels - 1]
	unlabeled_labels = np.eye(2)[unlabeled_labels - 1]

	# print(np.shape(labeled_votes), np.shape(unlabeled_votes))

	num_lab = np.shape(labeled_votes)[1]
	print("Unlab: " + str(num_unlab) + " | Lab: " + str(num_lab))

	train_labels = np.argmax(labeled_labels, axis=1)
	tl = np.argmax(unlabeled_labels, axis=1)

	if log_reg:
		import warnings
		warnings.filterwarnings('ignore')
		constraint_matrix, constraint_vector, constraint_sign = SG.compute_constraints_with_loss(SG.cross_entropy_linear, 
																								unlabeled_votes, 
																								labeled_votes, 
																								labeled_labels)


		# SET EPS here
		eps = 0.3
		L = 2 * np.sqrt(N + 1)
		squared_diam = 2
		T = int(np.ceil(L*L*squared_diam/(eps*eps)))
		h = eps/(L*L)
		T = 2000

		# transforming data w/ Resnet
		transformed_data = resnet_transform(unlabeled_X)
		initial_theta = np.random.normal(0, 0.1, (len(transformed_data[0]), C))

		model_theta = SG.subGradientMethod(transformed_data, constraint_matrix, constraint_vector, 
										constraint_sign, SG.cross_entropy_linear, SG.logistic_regression, 
										SG.projectToBall,initial_theta, 
										T, h, N, num_unlab, C, lr=True)



		c = eval_lr(transformed_data, tl, model_theta, C)
		print("Subgradient LR: " + str(c)) # acc ,  briar loss


	if sg:
		import warnings
		warnings.filterwarnings('ignore')
		constraint_matrix, constraint_vector, constraint_sign = SG.compute_constraints_with_loss(SG.Brier_loss_linear, 
																								unlabeled_votes, 
																								labeled_votes, 
																								labeled_labels)

		# SET EPS here
		eps = 0.3
		L = 2 * np.sqrt(N + 1)
		squared_diam = 2
		T = int(np.ceil(L*L*squared_diam/(eps*eps)))
		h = eps/(L*L)
		T = 2000

		model_theta = SG.subGradientMethod(unlabeled_votes, constraint_matrix, constraint_vector, 
										constraint_sign, SG.Brier_loss_linear, SG.linear_combination_labeler, 
										SG.projectToSimplex, np.array([1 / N for i in range(N)]), 
										T, h, N, num_unlab, C)
		
		# evaluate learned model
		c = eval_comb(unlabeled_votes, tl, model_theta)
		print("Subgradient: " + str(c)) # acc ,  briar loss

	if baseline:

		# majority vote
		mv_preds = []
		mv_probs = []

		for i in range(num_unlab):
		
			vote = np.zeros(C)
			for j in range(N):
				# vote_val = np.argmax(unlabeled_votes[j][i])
				# vote[vote_val] += 1

				vote += unlabeled_votes[j][i]
			mv_preds.append(np.argmax(vote))
			mv_probs.append(vote / N)

		mv_probs = np.array(mv_probs)
		mv_acc = np.mean(mv_preds == tl)
		
		print("MV: " + str(mv_acc))
		print("MV Briar:" + str(compute_avg_briar(tl, mv_probs, C)))

		# semi-supervised ds
		wl_votes_test = np.zeros((num_unlab, N))
		for i in range(N):
			wl_votes_test[:, i] = np.argmax(unlabeled_votes[i], axis=1)

		wl_votes_train = np.zeros((num_lab, N))
		for i in range(N):
			wl_votes_train[:, i] = np.argmax(labeled_votes[i], axis=1)

		wl_votes_test += 1
		wl_votes_train += 1
		train_labels += 1

		ds_model = SS.SemiSupervisedNaiveBayes(C, N)

		# create SS dataset
		votes = np.concatenate((wl_votes_train, wl_votes_test))
		labels = np.concatenate((train_labels, np.zeros(num_unlab))).astype(int)
		
		ds_model.estimate_label_model(votes, labels)
		ds_preds = ds_model.get_most_probable_labels(wl_votes_test)
		ds_probs = ds_model.get_label_distribution(wl_votes_test)
		
		ds_acc = np.mean(ds_preds == tl + 1)
		print("DS: " + str(ds_acc))

		ds_briar = compute_avg_briar(tl, ds_probs, C)
		print("DS Briar: " + str(ds_briar))

	if ind:
		print("Individual Weak Labelers")
		
		results_dict = {}

		for i in range(N):
			wl_preds = np.argmax(unlabeled_votes[i], axis=1)
			results_dict[i] = (np.mean(wl_preds == tl), compute_avg_briar(tl, unlabeled_votes[i], C))

		# sort list by lowest 0-1 acc
		sor_res = sorted(results_dict.items(), key=lambda x: -x[1][0])
		for i in range(3):
			print(sor_res[i])
	

	if pgmv:
		error_estimates = compute_errors(labeled_votes, train_labels)

		# convert votes to single output value
		ul_votes = np.argmax(unlabeled_votes, axis=2).T

		error_estimates, ul_votes, unlabeled_votes = correct_votes(error_estimates, ul_votes, unlabeled_votes)
		print(np.shape(ul_votes))

		# run PGMV
		best_sub, best_ep = PGMV.heuristic_algo1(1, error_estimates, ul_votes, 5, 9)
		probs, preds = eval_sub(unlabeled_votes, best_sub)

		print("PGMV: " + str((np.mean(preds == tl), compute_avg_briar(tl, probs, C))))

	if all_algo:
		error_estimates = compute_errors(labeled_votes, train_labels)
		ul_votes = np.argmax(unlabeled_votes, axis=2).T

		# transforming data w/ Resnet
		transformed_data = resnet_transform(unlabeled_X)
		# initial_theta = np.random.normal(0, 0.1, (len(transformed_data[0]), C))

		# run ALL
		probs, preds, labels = ALL.eval_all_lr(transformed_data, ul_votes.T, tl, transformed_data, tl, error_estimates)
		print(preds, labels)
		print("ALL: " + str((np.mean(preds == labels), compute_avg_briar(labels, probs, C))))


def domain_net_experiment(test_domain, sample, lab=100, ml=False, sg=False, baseline=False, ind=False, sup=False):
	'''
    运行DomainNet实验的函数。

    参数:
    test_domain - 目标域的索引。
    sample - 要评估的5个类别样本的数量。
    lab - 使用的标注数据的数量。
    ml - 是否运行最大似然法实验。
    sg - 是否运行次梯度法实验。
    baseline - 是否运行基线模型比较实验。
    ind - 是否独立评估每个弱标注器。
    sup - 是否运行监督学习实验。
    '''

	# 定义所有可能的域并从中移除测试域。
	domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
	domains.remove(test_domain)

	# 设置计算设备为CPU（对于需要GPU的实验，可以取消注释第一行并注释第二行）。
	# cuda0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	cuda0 = torch.device("cpu")

	# 加载测试域和样本的训练、验证和测试数据。
	trainloader, valloader, testloader = MDN.get_loaders(test_domain, sample)
	# 获取除测试域外的所有域上的弱标注器。
	wls = DWL.get_weak_labelers(sample, domains)
	# 获取测试域上训练的弱标注器。
	trained_wl = DWL.get_weak_labelers(sample, [test_domain])

	# N为弱标注器的数量（来自五个域），C为测试样本中的类别数量。
	N = len(wls)
	C = 5

	# 转换数据加载器以适应实验需求。
	trainloader, testloader = convert_loaders(trainloader, testloader)

	# 在训练和测试数据上应用弱标注器，得到标注和未标注数据的特征、投票和标签。
	labeled_X, labeled_votes, labeled_labels = DWL.apply_wls(wls, trainloader, cuda0, soft=True)
	unlabeled_X, unlabeled_votes, unlabeled_labels = DWL.apply_wls(wls, testloader, cuda0, soft=True)

	# 在测试数据上应用测试域的训练过的弱标注器。
	_, tc_votes, tc_tl = DWL.apply_wls(trained_wl, testloader, cuda0, soft=True)

	# 计算标注和未标注数据的数量。
	num_lab = np.shape(labeled_votes)[1]
	num_unlab = np.shape(unlabeled_votes)[1]

	# 从标注数据中随机选取一部分。
	lab_indices = random.sample(list(range(num_lab)), lab)

	# 使用选取的部分更新标注数据。
	labeled_X = labeled_X[lab_indices]  # 根据随机选取的索引更新标注数据的特征，以减小训练数据集的规模。
	labeled_votes = labeled_votes[:, lab_indices]  # 同样更新标注数据的弱标注器投票结果。
	labeled_labels = labeled_labels[lab_indices]  # 更新标注数据的标签。
	num_lab = np.shape(labeled_votes)[1]  # 重新计算更新后的标注数据数量。
	print("Unlab: " + str(num_unlab) + " | Lab: " + str(num_lab))  # 打印未标注数据和更新后的标注数据的数量。

	# 计算训练标签和测试标签。
	train_labels = np.argmax(labeled_labels, axis=1)  # 将标注数据的标签从one-hot编码转换为类别索引。
	tl = np.argmax(unlabeled_labels, axis=1)  # 将未标注数据的标签从one-hot编码转换为类别索引。
	tc_tl = np.argmax(tc_tl, axis=1)  # 将测试域的训练过的弱标注器的输出从one-hot编码转换为类别索引。
	print(np.shape(labeled_labels))  # 打印标注数据的标签维度信息。

	if sg:
		# 如果启用次梯度法实验。

		# 计算约束矩阵、约束向量和约束符号，这些是优化问题的一部分。
		constraint_matrix, constraint_vector, constraint_sign = SG.compute_constraints_with_loss(
			SG.Brier_loss_linear,  # 使用Brier损失函数作为计算损失的方法。
			unlabeled_votes,  # 未标注数据的投票结果。
			labeled_votes,  # 标注数据的投票结果。
			labeled_labels)  # 标注数据的实际标签。

		# 设置次梯度法的参数。
		eps = 0.3  # 学习率的调整参数。
		L = 2 * np.sqrt(N + 1)  # 根据弱标注器的数量计算的参数，用于控制步长。
		squared_diam = 2  # 假设的参数，可能与特征空间的直径有关。
		T = int(np.ceil(L * L * squared_diam / (eps * eps)))  # 计算迭代次数。
		h = eps / (L * L)  # 计算步长。
		T = 2000  # 重设迭代次数为固定值。

		# 调用次梯度方法进行优化。
		model_theta = SG.subGradientMethod(
			unlabeled_votes,  # 未标注数据的投票结果。
			constraint_matrix,  # 约束矩阵。
			constraint_vector,  # 约束向量。
			constraint_sign,  # 约束符号。
			SG.Brier_loss_linear,  # 使用Brier损失函数。
			SG.linear_combination_labeler,  # 标签组合的方法。
			SG.projectToSimplex,  # 投影到单纯形的方法，用于确保解的有效性。
			np.array([1 / C for i in range(C)]),  # 初始化解向量。
			T,  # 迭代次数。
			h,  # 步长。
			N,  # 弱标注器的数量。
			num_unlab,  # 未标注数据的数量。
			C)  # 类别的数量。

		# 评估学习到的模型。
		c = eval_comb(unlabeled_votes, tl, model_theta)
		print("Subgradient: " + str(c))  # 打印次梯度方法的评估结果，包括准确率和Brier损失。

	if baseline:
		# 如果启用基线实验。

		# 多数投票法。
		mv_preds = []  # 用于存储多数投票的预测结果。
		mv_probs = []  # 用于存储多数投票的概率。

		for i in range(num_unlab):
			# 对每个未标注数据进行投票。
			vote = np.zeros(C)  # 初始化投票计数器。
			for j in range(N):
				# 累加每个弱标注器的投票结果。
				vote += unlabeled_votes[j][i]
			mv_preds.append(np.argmax(vote))  # 选择得票最多的类别作为预测结果。
			mv_probs.append(vote / N)  # 计算每个类别的平均得票率作为预测概率。

		mv_probs = np.array(mv_probs)  # 转换为numpy数组。

		mv_acc = np.mean(mv_preds == tl)  # 计算多数投票法的准确率。
		print("MV: " + str(mv_acc))
		print("MV Briar:" + str(compute_avg_briar(tl, mv_probs, C)))  # 计算Brier分数。

		# 半监督学习的朴素贝叶斯。
		wl_votes_test = np.zeros((num_unlab, N))  # 初始化测试数据的弱标注器投票矩阵。
		for i in range(N):
			wl_votes_test[:, i] = np.argmax(unlabeled_votes[i], axis=1)  # 转换为类别索引。

		wl_votes_train = np.zeros((num_lab, N))  # 初始化训练数据的弱标注器投票矩阵。
		for i in range(N):
			wl_votes_train[:, i] = np.argmax(labeled_votes[i], axis=1)  # 转换为类别索引。

		wl_votes_test += 1  # 调整索引以适应朴素贝叶斯模型。
		wl_votes_train += 1
		train_labels += 1

		ds_model = SS.SemiSupervisedNaiveBayes(C, N)  # 初始化半监督学习模型。

		# 创建半监督学习数据集。
		votes = np.concatenate((wl_votes_train, wl_votes_test))  # 合并投票结果。
		labels = np.concatenate((train_labels, np.zeros(num_unlab))).astype(int)  # 合并标签，未标注数据标签为0。

		ds_model.estimate_label_model(votes, labels)  # 估计模型参数。
		ds_preds = ds_model.get_most_probable_labels(wl_votes_test)  # 获取测试数据的最可能标签。
		ds_probs = ds_model.get_label_distribution(wl_votes_test)  # 获取测试数据的标签分布。

		ds_acc = np.mean(ds_preds == tl + 1)  # 计算准确率。
		print("DS: " + str(ds_acc))

		ds_briar = compute_avg_briar(tl, ds_probs, C)  # 计算Brier分数。
		print("DS Briar: " + str(ds_briar))

	if sup:
		# 初始化一个预训练的弱标注器作为监督学习分类器。
		sup_classifier = DWL.WeakLabeler(C, pretrained=True)

		# 创建一个PyTorch DataLoader，仅包含标注数据，用于训练监督学习分类器。
		sup_loader = data.DataLoader(
			[(torch.tensor(labeled_X[i]), torch.tensor(np.argmax(labeled_labels[i]))) for i in range(num_lab)],
			shuffle=True, batch_size=100)
		# 训练分类器。
		DWL.train_weak_labeler(sup_classifier, sup_loader, valloader, cuda0)

		# 在测试数据上应用训练好的分类器，并计算准确率。
		_, sup_votes, sup_tl = DWL.apply_wls([sup_classifier], testloader, cuda0)
		print("Supervised: " + str(np.mean(np.argmax(sup_votes[0], axis=1) == tc_tl)))

	if ind:
		# 打印信息，表明以下是对每个弱标注器独立评估的结果。
		print("Individual Weak Labelers")

		# 遍历每个弱标注器，并在未标注数据上计算其准确率和Brier分数。
		for i in range(N):
			wl_preds = np.argmax(unlabeled_votes[i], axis=1)
			print(np.mean(wl_preds == tl), compute_avg_briar(tl, unlabeled_votes[i], C))

	if ml:
		# 使用最大似然方法估计标签的概率分布。
		val, x = ML.maximumLikelihood2(labeled_votes, labeled_labels, unlabeled_votes, ML.Brier_loss_linear)

		# 解析最大似然估计的结果，将其转换为未标注数据的预测概率分布。
		sol = np.zeros((num_unlab, C))
		for i in range(num_unlab):
			for j in range(C):
				sol[i, j] = x[i * C + j]

		# 根据预测的概率分布计算每个样本的预测类别，并计算准确率。
		ml_preds = np.argmax(sol, axis=1)
		ml_acc = np.mean(ml_preds == tl)
		print("Max Likelihood: " + str(ml_acc))


if __name__ == "__main__":
    # 初始化参数解析器
    parser = argparse.ArgumentParser()
    # 添加参数选项，用于决定是否运行DomainNet或AwA2实验
    parser.add_argument('--domainnet', default=True, type=str2bool, help="是否运行DomainNet或AwA2实验")
    # 添加参数选项，指定要运行实验的域
    parser.add_argument('--domain', default=1, type=int, help="运行脚本以计算弱标注器（wl）统计数据")
    # 添加参数选项，指定样本集的编号
    parser.add_argument('--sample', default=1, type=int, help="运行脚本以计算弱标注器（wl）统计数据")
    # 添加参数选项，决定是否与基线方法进行比较
    parser.add_argument('--baseline', default=False, type=str2bool, help="是否运行基线比较实验")
    # 添加参数选项，决定是否独立评估每个弱标注器
    parser.add_argument('--ind', default=False, type=str2bool, help="是否独立运行每个弱标注器")
    # 添加参数选项，决定是否运行最大似然方法
    parser.add_argument('--ml', default=False, type=str2bool, help="是否运行最大似然方法")
    # 添加参数选项，决定是否运行次梯度方法
    parser.add_argument('--sg', default=False, type=str2bool, help="是否运行次梯度方法")
    # 添加参数选项，决定是否运行监督学习方法
    parser.add_argument('--sup', default=False, type=str2bool, help="是否运行监督学习方法")
    # 添加参数选项，指定使用的标注数据的比例
    parser.add_argument('--num_lab', default=100, type=int, help="使用的标注数据的比例")
    # 添加参数选项，决定是否使用逻辑回归方法
    parser.add_argument('--log', default=False, type=str2bool, help="逻辑回归方法")

    # 为AwA2二分类案例添加PGMV和ALL基线方法的选项
    parser.add_argument('--pgmv', default=False, type=str2bool, help="PGMV二分类方法")
    parser.add_argument('--all', default=False, type=str2bool, help="对抗标签学习二分类方法")

    # 解析命令行参数
    args = parser.parse_args()

    # 固定随机种子，以确保实验可重复
	seed = 0
	np.random.seed(seed)  # 为NumPy的随机过程设置种子
	random.seed(0)  # 为Python内置的随机库设置种子
	torch.manual_seed(seed)  # 为PyTorch设置种子，影响CPU上的随机操作
	torch.cuda.manual_seed(seed)  # 为PyTorch的CUDA操作设置种子，影响GPU上的随机操作

	# 根据用户选择是否运行DomainNet实验或其他实验
	if args.domainnet:
		# 定义DomainNet数据集的所有可能域
		domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
		# 根据用户输入选择测试域
		test_domain = domains[args.domain - 1]
		# 打印选定的测试域和样本集编号
		print("Test Domain: " + test_domain)
		print("Sample: " + str(args.sample))
		# 调用domain_net_experiment函数运行DomainNet实验，传入用户指定的参数
		domain_net_experiment(test_domain, args.sample, lab=args.num_lab, ml=args.ml, baseline=args.baseline,
							  ind=args.ind, sup=args.sup, sg=args.sg)
	else:
		# 如果不是运行DomainNet实验，则运行另一个实验（可能是AwA2实验）
		# 调用aa2_experiment函数运行AwA2实验，传入用户指定的参数
		aa2_experiment(args.sample, lab=args.num_lab, baseline=args.baseline, ind=args.ind, sg=args.sg,
					   log_reg=args.log, pgmv=args.pgmv, all_algo=args.all)

