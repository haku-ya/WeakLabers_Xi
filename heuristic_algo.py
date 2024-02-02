import pulp as lp
import numpy as np
import itertools
import matplotlib.pyplot as plt
import math
import operator as op
from statistics import mean


def gen_bstring2(n, a):#递归函数，生成长度为n的所有二进制字符串。
    if(n==0):
        return [a.copy()]
    #当 n 等于 0 时，函数返回包含单个元素的列表，这个元素是 a 的一个副本。这是递归的终止条件，意味着不再需要进一步分解问题。
    else:
        a[n-1]=0
        l1 = gen_bstring2(n-1, a)
        #将列表 a 中的第 n-1 个元素设置为 0，然后递归地调用 gen_bstring2(n-1, a)。这将生成所有在第 n-1 位置为 0 的二进制字符串。
        a[n-1]=1
        l2 = gen_bstring2(n-1, a)
        #接着，将 a[n-1] 设置为 1，并再次递归地调用 gen_bstring2(n-1, a)。这次调用生成所有在第 n-1 位置为 1 的二进制字符串。
        return l1+l2

def gen_bstring(n):#使用gen_bstring2生成长度为n的所有二进制字符串。
    return gen_bstring2(n,np.zeros(n,dtype=np.int8))

def v_to_s(a):# 将一个由数字组成的列表转换为字符串。
    #它通过遍历列表中的每个元素，将每个数字转换为字符串，并将它们连接起来，形成一个单一的字符串。
    s = ""
    for x in a:
        s = s + str(x)
    return s

def flip_string(s):#翻转字符串中的字符，将每个 '0' 替换为 '1'，每个 '1' 替换为 '0'。
    output = ""
    for char in s:
        if char == "0":
            output = output + "1"
        else:
            output = output + "0"
    return output 

def majority_vote(vec): #计算向量中多数投票的结果（1或0）。
    if np.sum(vec) > len(vec)/2: 
        return 1
    else: 
        return 0

def build_vec_from_mat(idx, difference):#从给定的矩阵中构建向量。
    #它遍历 idx 中的每个索引对（i, j），只要 i < j，就从 difference 矩阵中选择相应的元素，并将其添加到结果向量中。
    ret = []
    for i in range(len(idx)):
        for j in range(i+1,len(idx)):
            ret.append(difference[idx[i]][idx[j]])
    return np.array(ret)

def build_vec(idx, epsilon):#根据索引列表和误差列表构建向量。
    #它遍历 idx 中的每个索引，并从 epsilon 中选择相应的元素添加到结果列表中。
    ret = []
    for i in range(len(idx)):
        ret.append(epsilon[idx[i]])
    return ret

def get_diff_from_votes(vote_data):#根据投票数据计算差异矩阵。#vote_data，一个二维列表，其中每一行代表一组投票数据。
    L = len(vote_data[0])#创建一个大小为 LxL 的零矩阵，其中 L 是 vote_data 中每行的长度。
    output = [[0 for i in range(L)] for j in range(L)]
    for a in range(len(vote_data)):
        for i in range(L):
            for j in range(L):
                if vote_data[a][i] != vote_data[a][j]:
                    output[i][j] += 1
    #对于 vote_data 的每一行，函数比较每一对元素。如果两个元素不同，对应的输出矩阵中的元素增加 1。
    for i in range(L):
        for j in range(L):
            output[i][j] = float(output[i][j])/len(vote_data)
    #函数将输出矩阵中的每个元素除以 vote_data 的行数，得到一个平均差异矩阵。
    return output

# Given n indices , prints a dict s.t., if the n labelers are ordered 
# by index, then the key is the sequence, and the val is the prob. 
# For example, the value in dict["011"] for 3 labelers 
# is the probability of seeing the sequence 0,1,1.
# def build_probs_from_votes(indices, votes):
#     N = len(indices)
#     possible = gen_bstring(N)
#     adict = {}
#     for a in possible:
#         if not majority_vote(a):
#             adict[v_to_s(a)] = 0
#     for i in range(len(votes)):
#         binary = ""
#         tmp = []
#         for j in indices:
#             binary += str(int(votes[i][j]))
#             tmp.append(votes[i][j])
#         if majority_vote(tmp):
#             adict[flip_string(binary)] = adict.get(flip_string(binary),0) + 1
#         else:
#             adict[binary] = adict.get(binary,0) + 1
#     s = len(votes)
#     for key in adict:
#         adict[key] = float(adict[key])/s
#     return adict
def build_probs_from_votes(indices, votes):#根据投票数据计算每个序列出现的概率。
    #indices（一个索引列表）和 votes（一个二维的投票数据数组）。
    N = len(indices)#每个投票序列中投票者的数量。
    possible = gen_bstring(N)#生成所有长度为 N 的二进制字符串（在这里表示可能的投票结果）。
    adict = {}#存储每种投票序列及其出现次数。
    for a in possible:
        if (a[0] == 0):
            adict[v_to_s(a)] = 0
    #遍历所有可能的二进制字符串，对于那些第一个元素为 0 的字符串，将它们的序列（转换为字符串格式）添加到 adict 中，并将其出现次数初始化为 0。
    for i in range(len(votes)):
        binary = ""
        tmp = []
        for j in indices:
            binary += str(int(votes[i][j]))
            tmp.append(votes[i][j])
        #遍历 votes 中的每一行（代表一次投票），并对每个索引 j 在 indices 中，将 votes[i][j] 转换为二进制字符串。
        if (votes[i][0] != 0):
            adict[flip_string(binary)] = adict.get(flip_string(binary),0) + 1
        else:
            adict[binary] = adict.get(binary,0) + 1
        #如果在当前行中第一个元素 votes[i][0] 不为 0，则在字典中更新翻转后的二进制字符串的出现次数；否则，更新原始二进制字符串的出现次数。
    s = len(votes)
    for key in adict:
        adict[key] = float(adict[key])/s
    #在统计完所有投票序列的出现次数后，通过将每个序列的出现次数除以总投票数 len(votes) 来计算每个序列的出现概率。
    return adict

def compute_upper_bound_old(data):  # 定义计算概率上界的函数
    eps = data[0] # 从输入数据中获取标注者的误差率列表
    diff = data[1] # 从输入数据中获取标注者之间的差异矩阵
    N = len(eps) # 计算标注者的数量
    possible_a = gen_bstring(N) # 生成长度为 N 的所有可能的二进制字符串列表

    # 为每个可能的二进制字符串定义线性规划变量 p_a 和 q_a
    p_a = [lp.LpVariable("p_"+v_to_s(a), 0, 1) for a in possible_a]  # p_a 变量，范围在 0 到 1 之间
    q_a = [lp.LpVariable("q_"+v_to_s(a), 0, 1) for a in possible_a]  # q_a 变量，范围在 0 到 1 之间

    # 创建一个线性规划问题，目标是最大化 q_a 变量的总和
    prob = lp.LpProblem("myProblem", lp.LpMaximize)  # 定义一个最大化问题
    prob += lp.lpSum([q_a[j] for j in range(len(possible_a))])  # 目标函数：q_a 的总和

    # 添加一个约束：p_a 和 q_a 变量的总和必须等于 1
    prob += lp.lpSum([q for q in q_a] + [p for p in p_a]) == 1

    # 为所有 p_a 和 q_a 变量添加非负约束
    for a in range(len(p_a)):
        prob += p_a[a] >= 0  # 确保 p_a[a] 不小于 0
        prob += q_a[a] >= 0  # 确保 q_a[a] 不小于 0

    # 约束 1：确保每个标注者 i 的误差率 eps[i] 等于相应的变量组合
    for i in range(N):
        # 对于每个标注者 i，计算两部分的和：
        # 第一部分是当标注者 i 的投票与多数投票不同的情况下，对应的 p_a 变量的和
        # 第二部分是当标注者 i 的投票与多数投票相同的情况下，对应的 q_a 变量的和
        # 这个总和应等于标注者 i 的误差率 eps[i]
        prob += lp.lpSum([p_a[j] for j in range(len(possible_a)) if possible_a[j][i] != majority_vote(possible_a[j])] +
                         [q_a[j] for j in range(len(possible_a)) if
                          possible_a[j][i] == majority_vote(possible_a[j])]) == eps[i]

    # 约束 2：处理标注者之间的差异
    x = 0  # 初始化索引变量 x
    for i in range(N):
        for j in range(N):
            if i < j:  # 只考虑 i < j 的情况，以避免重复计算
                # 计算标注者 i 和 j 在不同投票结果上的 p_a 和 q_a 变量的和
                # 这个和应等于 diff 矩阵中相应的差异值 diff[x]
                prob += lp.lpSum([p_a[k] for k in range(len(possible_a)) if possible_a[k][i] != possible_a[k][j]] +
                                 [q_a[k] for k in range(len(possible_a)) if possible_a[k][i] != possible_a[k][j]]) == \
                        diff[x]
                x += 1  # 更新索引变量 x

    # 解决线性规划问题
    status = prob.solve()

    # 根据解决方案的状态返回结果
    if lp.LpStatus[prob.status] == "Optimal":  # 如果找到最优解
        return prob.objective.value()  # 返回目标函数的值
    else:
        return 2  # 否则返回 2，表示没有找到最优解


def fft_metric(ind,epsilon,difference,k):
    # 计算给定指标的FFT度量

    # 获取指标对应的误差向量
    eps = build_vec(ind, epsilon)

    # 获取指标对应的差异向量
    diffs = build_vec_from_mat(ind, difference)

    # 获取指标的长度
    L = len(ind)

    # 计算FFT度量，即误差向量均值减去差异向量均值乘以常数 k
    output = mean(eps) - k * mean(diffs)

    # 返回计算结果
    return output


def compute_upper_bound_new(data):
    # 计算给定数据的概率上界

    eps = data[0]  # 标注者的误差率
    vote_probs = data[1]  # 差异概率矩阵
    N = len(eps)  # 标注者数量
    possible_a = gen_bstring(N)  # 所有可能的二进制字符串列表
    possible_a_str = []  # 将可能的二进制字符串转换为字符串形式
    for s in possible_a:
        possible_a_str.append(v_to_s(s))
    p_a = [lp.LpVariable("p_"+v_to_s(a), 0, 1) for a in possible_a]  # 创建线性规划变量 p_a
    q_a = [lp.LpVariable("q_"+v_to_s(a), 0, 1) for a in possible_a]  # 创建线性规划变量 q_a

    prob = lp.LpProblem("myProblem",lp.LpMaximize)  # 创建线性规划问题

    # 目标函数：最大化 q_a 的总和
    prob += lp.lpSum([q_a[j] for j in range(len(possible_a))])

    # 约束 3 和 4：确保 p_a 和 q_a 总和等于 1，同时限制变量的范围
    prob += lp.lpSum([q for q in q_a] + [p for p in p_a]) == 1  # 确保 q_a 和 p_a 总和等于1，这是线性规划问题的约束条件
    for a in range(len(p_a)):
        prob += p_a[a] >= 0  # 限制变量 p_a 的取值范围，必须大于等于0
        prob += q_a[a] >= 0  # 限制变量 q_a 的取值范围，必须大于等于0

    # 约束 1：确保每个标注者 i 的误差率 eps[i] 等于相应的变量组合
    for i in range(N):
        prob += lp.lpSum([p_a[j] for j in range(len(possible_a)) if possible_a[j][i] != majority_vote(possible_a[j])] +
                         [q_a[j] for j in range(len(possible_a)) if
                          possible_a[j][i] == majority_vote(possible_a[j])]) == eps[i]
    # 上述代码确保对于每个标注者 i，通过对 p_a 和 q_a 进行求和，得到的值等于该标注者的误差率 eps[i]。
    # 注意：p_a 和 q_a 的取值将在线性规划中计算以最大化目标函数，这些约束确保了这些变量与误差率之间的关系。

    # 约束 2：处理不同标注者之间的差异
    seen = []  # 用于跟踪已处理的字符串
    for j in range(len(possible_a)):
        if possible_a_str[j] not in seen:
            ind = possible_a_str.index(flip_string(possible_a_str[j]))  # 找到相反的字符串的索引
            if majority_vote(possible_a[j]):
                prob += lp.lpSum(p_a[j] + q_a[j] + p_a[ind] + q_a[ind]) == (vote_probs[possible_a_str[ind]])
            else:
                prob += lp.lpSum(p_a[j] + q_a[j] + p_a[ind] + q_a[ind]) == (vote_probs[possible_a_str[j]])
            seen.append(possible_a_str[j])
            seen.append(possible_a_str[ind])
    # 上述代码处理不同标注者之间的差异约束。它检查相反的二进制字符串是否已处理，
    # 如果没有处理过，则找到相反字符串的索引 ind，并确保相关变量的组合等于差异概率 vote_probs。
    # 这确保了不同标注者之间的差异在线性规划问题中得到了适当的处理。

    # 求解线性规划问题
    status = prob.solve()
    if lp.LpStatus[prob.status] == "Optimal":  # 检查线性规划问题是否有最优解
        return prob.objective.value()  # 返回最大化的目标函数值
    else:
        return 2  # 线性规划问题没有找到最优解，返回2表示未找到


def compute_upper_bound_new2(data):  # pair( epsilon, matrix) #另一种计算概率上界的方法。
    eps = data[0] # errors of labelers  # 标注者的误差率
    p = data[1] # differences  # 差异概率矩阵
    N = len(eps) # num labelers  # 标注者的数量
    possible = [a for a in gen_bstring(N) if (a[0] == 0)]  # 生成可能的标签组合，要求第一个标签为0
    c_a = [lp.LpVariable("c_"+v_to_s(a), 0, 1) for a in possible]  # 创建线性规划变量 c_a，取值范围为 [0, 1]

    prob = lp.LpProblem("myProblem",lp.LpMaximize)  # 创建线性规划问题，最大化问题
    # Objective function
    prob += lp.lpSum([p[v_to_s(possible[j])]*c_a[j] for j in range(len(possible))])  # 目标函数，计算概率上界

    # Constraint 1
    for i in range(N):
        prob += lp.lpSum([p[v_to_s(possible[j])]*c_a[j] for j in range(len(possible)) if possible[j][i] == majority_vote(possible[j])] +
                         [p[v_to_s(possible[j])]*(1-c_a[j]) for j in range(len(possible)) if possible[j][i] != majority_vote(possible[j])]) <= eps[i]
    # 约束条件1：确保每个标注者 i 的误差率 eps[i] 小于等于相应的变量组合

    status = prob.solve()  # 解决线性规划问题
    if lp.LpStatus[prob.status] == "Optimal":
        return prob.objective.value()  # 返回线性规划的目标函数值（概率上界）
    else:
        return 2  # LP Terminated without finding optimal solution  # 如果线性规划没有找到最优解，返回2表示未找到最优解

# 根据给定的参数选择一组标注者，以最小化某种度量。
def heuristic_algo1(algo, epsilon, votes_matrix, min_labelers, size):
    votes = votes_matrix  # 投票矩阵的副本
    # 计算投票矩阵中各标注者之间的差异度
    difference = get_diff_from_votes(votes)

    # 纠正错误率大于1/2的标注者，即他们的标记几乎总是错误的，通过反转其投票来“纠正”它们
    for i in range(len(epsilon)):
        if(epsilon[i] > 1/2):
            for k in range(len(votes)):
                votes[k][i] = 1 - votes[k][i]  # 反转投票
            for j in range(len(epsilon)):
                if j != i:
                    # 也需要更新差异度矩阵，反映投票的反转
                    difference[i][j] = 1 - difference[i][j]
                    difference[j][i] = 1 - difference[j][i]
            epsilon[i] = 1 - epsilon[i]  # 更新错误率

    # 根据选择的算法（algo）决定使用哪个数据集
    if algo == 1:
        data = difference
    else:
        data = votes

    labelers = list(range(len(epsilon)))  # 所有标注者的索引
    best_ep = 1  # 初始化最佳误差率为1，即最差情况
    best_S = 0  # 最佳标注者子集初始化

    for j in range(len(labelers)):
        # 对每个标注者尝试构建最优子集
        S = [j]  # 当前考虑的标注者子集
        tmp_ep = 2  # 临时误差率，初始化为一个大于可能的最大误差率的值

        # 循环直到子集达到最小标注者数
        while len(S) < min_labelers:
            tmp_ep = 2  # 重新初始化临时误差率
            add_S = []  # 准备添加到S的标注者
            unused = labelers.copy()  # 未使用的标注者

            # 移除已经被选中的标注者
            for x in S:
                unused.remove(x)

            # 计算未使用的标注者组合，并尝试找到最优的两个标注者加入S
            combs = list(itertools.combinations(unused, 2))
            for c in combs:
                tmp = S.copy()
                for ind in range(len(c)):
                    tmp.append(c[ind])
                tmp.sort()
                # 根据算法计算误差率上界
                if algo == 1:
                    x = compute_upper_bound_old((build_vec(tmp, epsilon), build_vec_from_mat(tmp, data)))
                else:
                    x = compute_upper_bound_new2((build_vec(tmp, epsilon), build_probs_from_votes(tmp, data)))

                # 更新临时误差率和待添加标注者
                if x < tmp_ep:
                    tmp_ep = x
                    add_S = c

            # 如果没有找到可以添加的标注者，跳出循环
            if len(add_S) == 0:
                break
            else:
                # 将找到的标注者添加到S中
                for ind in range(len(add_S)):
                    S.append(add_S[ind])
                S.sort()
        # skipping if no optimal solution is found
        # 如果找到的子集不满足最小标注者数量要求，跳过当前循环
        if len(S) < min_labelers:
            print("skipping")
            continue
        #继续寻找最佳标注者子集的循环，直到达到指定的size大小或找不到更好的组合为止：

        previous_ep = tmp_ep
        curr_ep = tmp_ep

        while True:
            # 如果当前子集的大小已经达到用户指定的最大大小，检查是否需要更新最佳子集
            if len(S) == size:
                if curr_ep < best_ep:
                    best_ep = curr_ep
                    best_S = S
                break  # 退出循环

            add_S = 0
            unused = labelers.copy()
            # 移除已经被选中的标注者
            for x in S:
                unused.remove(x)
            # 如果未使用的标注者少于2个，停止尝试添加
            if len(unused) < 2: break

            # 计算所有可能的标注者对组合
            combs = list(itertools.combinations(unused, 2))
            for c in combs:
                tmp = S.copy()
                for ind in range(len(c)):
                    tmp.append(c[ind])
                tmp.sort()

                # 根据选定算法计算误差率上界
                if algo == 1:
                    x = compute_upper_bound_old((build_vec(tmp, epsilon), build_vec_from_mat(tmp, data)))
                else:
                    x = compute_upper_bound_new2((build_vec(tmp, epsilon), build_probs_from_votes(tmp, data)))

                # 如果找到了更低的误差率，更新当前最佳添加的标注者
                if x < curr_ep:
                    curr_ep = x
                    add_S = c

                    # 如果当前步骤找到的误差率低于之前的，更新并继续寻找
            if curr_ep < previous_ep:
                previous_ep = curr_ep
                for ind in range(len(add_S)):
                    S.append(add_S[ind])
                S.sort()
            else:
                # 如果没有找到更好的，检查是否需要更新最佳子集，然后退出循环
                if curr_ep < best_ep:
                    best_ep = curr_ep
                    best_S = S
                break

            # 返回最佳标注者子集和对应的误差率
        return (best_S, best_ep)


#num is the number of labelers needed, delta is the scaling param
#in the algorithm 
# 另一种启发式算法，根据给定的参数选择一组标注者。
def heuristic_algo2(epsilon, votes_matrix, num, delta):
    # 对投票矩阵进行预处理，将投票值从基于1改为基于0
    votes = votes_matrix - 1

    # 计算投票矩阵中各标注者之间的差异度
    difference = get_diff_from_votes(votes)

    # 纠正错误率大于1/2的标注者
    for i in range(len(epsilon)):
        if(epsilon[i] > 1/2):
            for k in range(len(votes)):
                votes[k][i] = 1 - votes[k][i]  # 反转投票
            for j in range(len(epsilon)):
                if j != i:
                    # 更新差异度矩阵，反映投票的反转
                    difference[i][j] = 1 - difference[i][j]
                    difference[j][i] = 1 - difference[j][i]
            epsilon[i] = 1 - epsilon[i]  # 更新错误率

    L = len(epsilon)  # 标注者总数
    # 生成所有可能的标注者组合，组合大小为num
    combs = list(itertools.combinations(range(L), num))
    min_dist = 99999  # 初始化最小距离（或度量）为一个很大的数
    best_c = (0,0,0)  # 初始化最佳组合

    # 遍历所有组合，寻找最优解
    for c in combs:
        x = fft_metric(c, epsilon, difference, delta)  # 计算当前组合的度量
        if x < min_dist:
            min_dist = x  # 更新最小距离
            best_c = c  # 更新最佳组合

    # 返回最小度量和对应的最佳标注者组合
    return(min_dist, best_c)
