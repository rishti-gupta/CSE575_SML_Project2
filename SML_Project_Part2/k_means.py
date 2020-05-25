import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import pylab


# Function to calculate the euclidean distance between two points
def euc_dist(x, y):
    tmp = np.square(np.subtract(x, y), dtype=np.float64)
    dist = math.sqrt(tmp[0] + tmp[1])
    return dist


# k-means using strategy-1
def k_means_strategy_1(n, k):
    centroids_arr = []  # creating a centroid array which stores all of them
    centroid_num = list('0')  # stores the no. of the cluster for each data point
    prev = list()  # To store the no. of cluster for each data point in the previous iteration
    totalNumber = 0  # total number of iterations

    # Initialising centroids at random (in k clusters)
    offset = k
    while offset > 0:
        r = np.random.randint(0, 300)
        centroids_arr.append(n[r])
        offset -= 1

    # checking if k-means has reached its convergence
    # till the time when centroid value does not change or total number has been achieved
    while prev != centroid_num or totalNumber == 500:
        clusters = {k: [] for k in range(1, k + 1)}
        prev = centroid_num.copy()
        centroid_num = []
        # Assigning the data points to k clusters
        for offset in n:
            dist_list = []
            for offset1 in centroids_arr:
                dist = euc_dist(offset, offset1)
                # list to store the distance between data-points and centroids
                dist_list.append(dist)

                # finding the minimum of the distance
                # create the corresponding clusters
            clusters[dist_list.index(min(dist_list)) + 1].append(n.index(offset))
            # store the cluster numbers of each data point in a list
            centroid_num.append(dist_list.index(min(dist_list)) + 1)

            # Updating the k centroids with the new values
            for p in range(1, k + 1):
                data_point = clusters[p]  # forming the corresponding cluster for the data points
                for i in data_point:
                    meanList = list()
                    for d in range(len(n[i])):
                        L = list()
                        # appending the data points in the list and calculating the mean of all the points
                        L.append(n[i][d])
                        meanList.append(np.mean(L))
                    centroids_arr[p - 1] = meanList
            totalNumber += 1
            save = clusters
        return save, centroids_arr


def k_means_strategy_2(n, k):
    centroids_arr = []  # creating a centroid array which stores all of them
    centroid_num = list('0')  # stores the no. of the cluster for each data point
    prev = list()  # To store the cluster numbers for each data point in the previous iteration
    totalNumber = 0  # total number of iterations

    # Initialize k cluster centroids based on strategy-2
    r = np.random.randint(0, 300)
    centroids_arr.append(n[r])
    x = 1
    while x != k - 1:
        dist_list = list()
        for offset in n:
            s = 0
            for offset1 in centroids_arr:
                dist = euc_dist(offset, offset1)
                s += dist
            s /= len(centroids_arr)
            dist_list.append(s)
        M = dist_list.index(max(dist_list))
        centroids_arr.append(n[M])
        x += 1

    # Check for k-means convergence criteria
    while prev != centroid_num or totalNumber == 500:
        clusters = {k: [] for k in range(1, k + 1)}
        prev = centroid_num.copy()
        centroid_num = []
        # Assignment of data points to k clusters
        for offset in n:
            dist_list = []
            for offset1 in centroids_arr:
                dist = euc_dist(offset, offset1)
                # list to store the distance between data-points and centroids
                dist_list.append(dist)

                # find minimum of the distance and create clusters
            clusters[dist_list.index(min(dist_list)) + 1].append(n.index(offset))
            # store the cluster numbers of each data point in a list
            centroid_num.append(dist_list.index(min(dist_list)) + 1)

        # Update k centroids with new values
        for p in range(1, k + 1):
            data_point = clusters[p]
            for i in data_point:
                meanList = list()
                for d in range(len(n[i])):
                    L = list()
                    L.append(n[i][d])
                    meanList.append(np.mean(L))
                centroids_arr[p - 1] = meanList
        totalNumber += 1
        save = clusters
    return save, centroids_arr


# defining the objective function
def objective_fn(data, cluster_data, centroids):
    obj_func = 0
    for offset in cluster_data:
        for offset1 in cluster_data[offset]:
            d = np.square(euc_dist(data[offset1], centroids[offset - 1]))
            obj_func = obj_func + d
    return obj_func


'''
    Run K-means for k= 2 to 10 using  k_means Strategy-1
'''
# The Matlab data has been converted to csv file and the following command reads that csv file
dataset = pd.read_csv('Dataset.csv', names=["X", "Y"])

# plotting the objective function graph using strategy-1
k_means_data = list()
data_temp = dataset.copy()
data_tempX = data_temp['X'].tolist()
data_tempY = data_temp['Y'].tolist()
for i in range(len(data_tempX)):
    dt = list()
    dt.append(data_tempX[i])
    dt.append(data_tempY[i])
    k_means_data.append(dt)

# Initialization-1 using Strategy-1
opt_k = []
for k in range(2, 11):
    cluster_data, centroids = k_means_strategy_1(k_means_data, k)

    opt = objective_fn(k_means_data, cluster_data, centroids)
    opt_k.append(opt)

# Initialization-2 using Strategy-1
opt_k1 = []
for k in range(2, 11):
    cluster_data, centroids = k_means_strategy_1(k_means_data, k)

    opt = objective_fn(k_means_data, cluster_data, centroids)
    opt_k1.append(opt)

'''
    Task 2- Run K-means for k= 2 to 10 using Strategy-2
'''

# Initialization-1 using Strategy-2
opt_k2 = []
for k in range(2, 11):
    cluster_data, centroids = k_means_strategy_2(k_means_data, k)

    opt = objective_fn(k_means_data, cluster_data, centroids)
    opt_k2.append(opt)

# Initialization-2 using Strategy-2
opt_k3 = []
for k in range(2, 11):
    cluster_data, centroids = k_means_strategy_2(k_means_data, k)

    opt = objective_fn(k_means_data, cluster_data, centroids)
    opt_k3.append(opt)

# plotting both the objective functions with each having two initializations
# building a figure with two subplots stacked horizontally
fig, (ax1, ax2) = plt.subplots(1, 2, num='K-Means Clustering (Project Part-2)')
fig.suptitle('K-Means Clustering')
fig.subplots_adjust(wspace=0.5)

# subplot 1 for strategy-1
ax1.plot(range(2, 11), opt_k, '-b')
ax1.plot(range(2, 11), opt_k1, '-r')
ax1.legend(('Initialization-1', 'Initialization-2'), loc='upper right', fontsize='x-small')
ax1.set_xlabel("Number of Clusters(k)")
ax1.set_ylabel("Objective Function(J)")
ax1.set_title("k-means using Strategy-1")

# subplot 2 for strategy-1
ax2.plot(range(2, 11), opt_k2, '-g')
ax2.plot(range(2, 11), opt_k3, '-y')
ax2.legend(('Initialization-1', 'Initialization-2'), loc='upper right', fontsize='x-small')
ax2.set_xlabel("Number of Clusters(k)")
ax2.set_ylabel("Objective Function(J)")
ax2.set_title("k- means using Strategy-2")

# display the plot
plt.show()
