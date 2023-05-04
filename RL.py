from gurobipy import GRB
import gurobipy as gp
import pandas as pd
import random
from copy import deepcopy
import numpy as np
import math
from collections import namedtuple
from typing import List
from tqdm import tqdm
from sys import exit
import os
import time
from time import process_time
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib
import os.path


def readData(filename, n):
    stream = ""
    with open(filename, "r") as file:
        stream = file.readlines()
    if stream == "":
        print("Error in reading file")
    else:
        print("Read file", filename)

    vehicleNumber, capacity = [int(i) for i in stream[4].split()]
    # print(vehicleNumber, capacity)
    fields = ("CUST-NO.", "XCOORD.", "YCOORD.", "DEMAND", "READY-TIME", "DUE-DATE", "SERVICE-TIME")
    data = list()
    for i in range(9, len(stream)):
        if stream[i] == "\n":
            continue
        val = stream[i].split()
        if len(val) != len(fields):
            print("Error in reading data")
            continue
        customer = dict(zip(fields, val))
        data.append(customer)

    # Consider only depot + 50 customers
    data = data[0:n + 1]
    data.append(data[0])  # The depot is represented by two identical
    # nodes: 0 and n+1
    data[-1]["CUST-NO."] = "51"
    x = []
    y = []
    q = []
    a = []
    b = []
    for customer in data:
        x.append(int(customer["XCOORD."]))
        y.append(int(customer["YCOORD."]))
        q.append(int(customer["DEMAND"]))
        a.append(int(customer["READY-TIME"]))
        b.append(int(customer["DUE-DATE"]))

    return vehicleNumber, capacity, x, y, q, a, b
    '''
    routes = []
    with open(filename) as f:
        for line in f:
            a = line.split(" ")
    return routes
    '''


# create distance matrix
def createDistanceMatrix(x, y):
    n = len(x)  # number of customers
    d = np.zeros((n, n))  # distance matrix
    for i in range(n):
        for j in range(i + 1, n):
            p1 = np.array([x[i], y[i]])
            p2 = np.array([x[j], y[j]])
            d[i, j] = d[j, i] = int(round(np.linalg.norm(p1 - p2)))
    return d


def computeImpact(IS, IU, LD, feasiblePositions):
    IR = sum(LD) / len(feasiblePositions)
    bestImpact = None
    bestPosition = None
    for i in range(len(feasiblePositions)):
        impact = IS[i] + IU[i] + IR
        if bestPosition is None or impact < bestImpact:
            bestImpact = impact
            bestPosition = feasiblePositions[i]

    return bestPosition, bestImpact


def computeRouteCost(route, d):
    cost = 0
    for i in range(len(route) - 1):
        cost += d[route[i], route[i + 1]]
    return cost


def computeISIULD(posU, route, arr, s, a, b, d, Jminu):
    b1 = b2 = b3 = bs = be = br = 1 / 3
    u = route[posU]
    posI = posU - 1
    posJ = posU + 1
    i = route[posI]
    j = route[posJ]
    IS = arr[posU] - a[u]
    IU = 1 / (max([len(Jminu), 1])) * sum([max([b[n] - a[u] - d[u, n], b[u] - a[n] - d[u, n]]) for n in Jminu])
    c1 = (d[i, u] + d[u, j] - d[i, j])
    c2 = ((b[j] - (arr[posI] + d[i, j])) - (b[j] - (arr[posU] + d[i, j])))
    c3 = (b[u] - (arr[posI] + d[i, u]))
    LD = b1 * c1 + b2 * c2 + b3 * c3

    return IS, IU, LD


def addRoutesToMaster(routes, mat, costs, d):  # routes is the set of all route, set of lists
    for i in range(len(routes)):
        cost = d[routes[i][0], routes[i][1]]
        for j in range(1, len(routes[i]) - 1):  # calculate the travelling cost
            cost += d[routes[i][j], routes[i][j + 1]]
            mat[routes[i][j] - 1, i] += 1  # if the customer number is 4, then it should be in 3 row (0, 1, 2, 3)
        costs[i] = cost


def insertNode(route, node, position, s, arr, d, a):
    newRoute = route[:]
    newRoute.insert(position, node)
    newS = [];
    newArr = []
    for i in range(position):
        newS.append(s[i])
        newArr.append(arr[i])
    for i in range(position, len(newRoute)):
        newArr.append(newS[i - 1] + d[newRoute[i - 1], newRoute[i]])
        newS.append(max(newArr[i], a[i]))
    return newRoute, newS, newArr


def routeIsFeasible(route, a, b, s, d, q, Q):
    cap = sum([q[node] for node in route])
    if cap > Q:
        return False
    for i in range(len(route)):
        if not ((s[i] >= a[route[i]]) and (s[i] <= b[route[i]])):
            return False
    return True


def reduceTimeWindows(n, d, readyt, duedate):
    a = readyt[:]
    b = duedate[:]
    update = True
    while update:
        update = False
        for k in range(1, n + 1):
            # Ready Time
            minArrPred = min([b[k], min([a[i] + d[i, k] for i in range(n + 1) if i != k])])
            minArrNext = min([b[k], min([a[j] - d[k, j] for j in range(1, n + 2) if j != k])])
            newa = int(max([a[k], minArrPred, minArrNext]))
            if newa != a[k]:
                update = True
            a[k] = newa

            # Due date
            maxDepPred = max([a[k],
                              max([b[i] + d[i, k] for i in range(n + 1) if i != k])])
            maxDepNext = max([a[k],
                              max([b[j] - d[k, j] for j in range(1, n + 2) if j != k])])
            newb = int(min([b[k], maxDepPred, maxDepNext]))
            if newb != b[k]:
                update = True
            b[k] = newb
    return a, b


def initializePathsWithImpact(d, n, a, b, q, Q):
    J = list(range(1, n + 1))
    routes = []
    costs = []

    while J:
        # Find the furthest node from depot in J and initialize route with it
        far = -1
        max_dist = -1
        for j in J:
            if d[0, j] > max_dist:
                far = j
                max_dist = d[0, j]

        route = [0, far, n + 1]
        arr = [0, d[0, far]]
        s = [0, max([a[far], arr[1]])]
        arr.append(s[1] + d[far, n + 1])
        s.append(max(arr[2], a[n + 1]))
        J.remove(far)

        feasible = J[:]

        while feasible:
            proposals = dict()
            for u in feasible:
                bestImpact = None
                bestPosition = None
                Jminu = J[:]
                Jminu.remove(u)
                feasiblePositions = []
                IS = IU = LD = []
                for pos in range(1, len(route)):
                    newRoute, newS, newArr = insertNode(route, u, pos, s, arr, d, a)
                    if routeIsFeasible(newRoute, a, b, newS, d, q, Q):
                        feasiblePositions.append(pos)
                        Is, Iu, Ld = computeISIULD(pos, newRoute, newArr, newS, a, b, d, Jminu)
                        IS.append(Is)
                        IU.append(Iu)
                        LD.append(Ld)

                if not feasiblePositions:
                    feasible.remove(u)
                else:
                    bestPosition, bestImpact = computeImpact(IS, IU, LD, feasiblePositions)
                    proposals[bestImpact] = (u, bestPosition)
                # END FOR
            if proposals:
                nodeToInsert, insertPos = proposals[min(list(proposals.keys()))]
                route, s, arr = insertNode(route, nodeToInsert, insertPos, s, arr, d, a)
                feasible.remove(nodeToInsert)
                J.remove(nodeToInsert)
            # END WHILE

        routes.append(route)
        costs.append(computeRouteCost(route, d))
        # END WHILE
    return routes


# BipartiteGNN class: GNN model for the bipartite graph
# __init__: initialization
# buildGNN: sets the input shapes. Called during initialization
# callGNN: an input is a tuple containing the three matrices, output is logit vector for the variables nodes
# train_or_test:
class BipartiteGNN(tf.keras.Model):
    '''
    Initialization of the different modules and attributes:
    - embedding_size : Embedding size for the intermediate layers of the neural networks
    - cons_num_features : Number of constraint features, the constraints data matrix expected has the shape (None,cons_num_features)
    - vars_num_features : Number of variable features, the variables data matrix expected has the shape (None,vars_num_features)
    - learning_rate : Optimizer learning rate
    - activation : Activation function used in the neurons
    - initializer : Weights initializer
    '''

    def __init__(self, embedding_size=32, cons_num_features=2,
                 vars_num_features=9, learning_rate=1e-3, seed=0,
                 activation=tf.keras.activations.relu, initializer=tf.keras.initializers.Orthogonal):
        self.seed_value = seed
        tf.random.set_seed(self.seed_value)
        super(BipartiteGNN, self).__init__()

        self.embedding_size = embedding_size
        self.cons_num_features = cons_num_features
        self.vars_num_features = vars_num_features
        self.learning_rate = learning_rate
        self.activation = activation

        self.initializer = initializer(seed=self.seed_value)
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

        # constraints embedding layer
        self.cons_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.embedding_size, activation=self.activation,
                                  kernel_initializer=self.initializer),
        ])

        # columns embedding layer
        self.var_embedding = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.embedding_size, activation=self.activation,
                                  kernel_initializer=self.initializer),
        ])

        # NN responsible for the intermediate updates
        self.join_features_NN = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.embedding_size, activation=self.activation,
                                  kernel_initializer=self.initializer),
            tf.keras.layers.Dense(units=self.embedding_size, activation=self.activation,
                                  kernel_initializer=self.initializer)
        ])

        # Representations updater for the constraints, called after the aggregation
        self.cons_representation_NN = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.embedding_size, activation=self.activation,
                                  kernel_initializer=self.initializer),
        ])
        # Representations updater for the columns, called after the aggregation
        self.vars_representation_NN = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.embedding_size, activation=self.activation,
                                  kernel_initializer=self.initializer),
        ])

        # NN for final output, i.e., one unit logit output
        self.output_module = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.embedding_size, activation=self.activation,
                                  kernel_initializer=self.initializer),
            tf.keras.layers.Dense(units=self.embedding_size, activation=self.activation,
                                  kernel_initializer=self.initializer),
            tf.keras.layers.Dense(units=1, activation=None, kernel_initializer=self.initializer)
        ])

        # Build of the input shapes of all the NNs
        self.buildGNN()

        # Order set for loading/saving the model, store weight and bias
        self.variables_topological_order = [v.name for v in self.variables]
        # print(self.variables_topological_order)
        # exit(0)

    '''
    Build function, sets the input shapes. Called during initialization
    '''

    def buildGNN(self, **kwargs):
        self.cons_embedding.build([None, self.cons_num_features])
        self.var_embedding.build([None, self.vars_num_features])
        self.join_features_NN.build([None, self.embedding_size * 2])
        self.cons_representation_NN.build([None, self.embedding_size * 2])
        self.vars_representation_NN.build([None, self.embedding_size * 2])
        self.output_module.build([None, self.embedding_size])
        self.built = True

    '''
    Main function taking as an input a tuple containing the three matrices :
    - cons_features : Matrix of constraints features, shape : (None, cons_num_features)
    - edge_indices : Edge indices linking constraints<->variables, shape : (2, None)
    - vars_features : Matrix of variables features, shape : (None, vars_num_features)
    Output : logit vector for the variables nodes, shape (None,1)
    '''

    def callGNN(self, inputs, training=None, mask=None):
        # print("The code is running", inputs[0].shape)
        cons_features, edge_indices, vars_features = inputs
        # print("#######################")
        # print(cons_features)
        # print(edge_indices)
        # print(vars_features)
        # print("#######################")
        # Nodes embedding, constraints and variables
        cons_features = self.cons_embedding(cons_features)
        vars_features = self.var_embedding(vars_features)

        # ==== First Pass : Variables -> Constraints ====
        # compute joint representations
        joint_features = self.join_features_NN(
            tf.concat([
                tf.gather(
                    cons_features,
                    axis=0,
                    indices=edge_indices[0])
                ,
                tf.gather(
                    vars_features,
                    axis=0,
                    indices=edge_indices[1])
                ### change this number to edge weights (patterns)
            ], 1)
        )

        # Aggregation step
        output_cons = tf.scatter_nd(
            updates=joint_features,
            indices=tf.expand_dims(edge_indices[0], axis=1),
            shape=[cons_features.shape[0], self.embedding_size]
        )

        # Constraints representations update
        output_cons = self.cons_representation_NN(tf.concat([output_cons, cons_features], 1))

        # ==== Second Pass : Constraints -> Variables ====
        # compute joint representations
        joint_features = self.join_features_NN(
            tf.concat([
                tf.gather(
                    output_cons,
                    axis=0,
                    indices=edge_indices[0])
                ,
                tf.gather(
                    vars_features,
                    axis=0,
                    indices=edge_indices[1])
            ], 1)
        )

        # Aggregation step
        output_vars = tf.scatter_nd(
            updates=joint_features,
            indices=tf.expand_dims(edge_indices[1], axis=1),
            shape=[vars_features.shape[0], self.embedding_size]
        )

        # Variables representations update
        output_vars = self.vars_representation_NN(tf.concat([output_vars, vars_features], 1))

        # ==== Final output from the variables representations (constraint nodes are ignored)
        output = self.output_module(output_vars)
        return output

    '''
    Training/Test function, Input: 
    - data : a batch of data, type : tf.data.Dataset
    - train: boolean, True if function called for training (i.e., compute gradients and update weights),
                False if called for test
    Output: tuple(Loss, Accuracy, Recall, TNR) : Metrics
    '''

    def save_state(self, path):
        import pickle
        with open(path, 'wb') as f:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                pickle.dump(v.numpy(), f)

    '''
    Load an existing model from a given path
    '''

    def restore_state(self, path):
        import pickle
        with open(path, 'rb') as f:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                v.assign(pickle.load(f))

    # output is mean_loss
    def train_or_test(self, data, labels, totals_0, actions_0, action_info, train=False):
        mean_loss = 0
        batches_counter = 0

        for batch in data:
            cons_features, edge_indices, vars_features = batch
            input_tuple = (cons_features, edge_indices, vars_features)

            total_0 = totals_0[batches_counter]

            label = labels[batches_counter]

            # When called train=True, compute gradient and update weights
            if train:
                with tf.GradientTape() as tape:
                    # Get logits from the bipartite GNN model

                    logits = self.callGNN(input_tuple)
                    # print(total_0)
                    # do not count the loss from the nodes already in the basis
                    label[0:-total_0[0]] = logits[0:-total_0[0]]

                    # should not be mean_squared_error as it's then scaled down by number of nodes
                    loss = tf.keras.metrics.mean_squared_error(label, logits)
                    # loss = (loss * label.shape[0]) / total_0[0]  ## this is a quick fix, as there are far less action nodes compared to
                    loss = (loss * label.shape[0])

                # Compute gradient and update weights
                grads = tape.gradient(target=loss, sources=self.variables)
                self.optimizer.apply_gradients(zip(grads, self.variables))

            # If no optimizer instance set, no training is performed, give outputs and metrics only
            else:
                logits = self.callGNN(input_tuple)
                loss = tf.keras.metrics.mean_squared_error(label, logits)

            loss = tf.reduce_mean(loss)

            # Batch loss, accuracy, confusion matrix
            mean_loss += loss
            batches_counter += 1
            # confusion_mat += confusion_matrix(labels, prediction)

        # Batch average loss
        mean_loss /= batches_counter
        return mean_loss


class VRP(object):
    def __init__(self, INSTANCE_NAME, n):  # input the name of that instance
        # each instance corresponds to self.state = self.env.reset() in learning_method in agent.py
        # state: current graph (connection, current column nodes, current constraint node and their features)
        #### static info: problem definition, same info used for initialization this instance

        Kdim, Q, x, y, q, a, b = readData(INSTANCE_NAME, n)

        self.n = n  # number of customers
        self.Kdim = Kdim  # number of vehicles
        self.Q = Q  # capacity of vehicles
        self.x = x  # x coordinate
        self.y = y  # y coordinate
        self.q = q  # demand of customers
        self.a = a  # earliest time to serve
        self.b = b  # latest time to serve
        self.d = createDistanceMatrix(x, y)  # distance matrix

        # dynamic info for building RMP, PP
        # update by addRoutetoMaster
        # None represents the absence of a value.
        self.A = None  # like self.patterns as before
        self.c = None  # routes costs
        self.routes = None  # route list storing current routes

        # dynamic info (info needed to for state + reward), get from CG iterations from solving current RMP and PP:
        self.objVal_history = []
        self.total_steps = 0

        # action with their reduced cost (stored as tuple) ([all the patterns],[(data for those routes)])
        self.available_action = ()
        self.count_convergence = 0

        '''  
        Info for column and constraint node features, stored using list, length will change: 
        1. column: number of constraint participation, current solution value (if not in the basis, 0 -> int or not),
        enter, leave, RC, cost, # in basis, # not in basis, # of customers in the route
        2. constraint : dual (shadow price), number of columns contributing to the constraint
        '''
        # for all the columns (size fixed)
        self.In_Cons_Num = []  # feature 0
        self.ColumnSol_Val = []  # feature 1
        self.ColumnIs_Basic = []
        # just left the basis in last iteration, 0 not just left
        self.just_left = []  # feature 2
        self.just_enter = []  # feature 3
        # for all the variable that are in the basis, count the number of times it's in basis, otherwise 0
        self.stay_in = []  # feature 6
        self.stay_out = []  # feature 7

        # for all the constraints (size fixed)
        self.Shadow_Price = None
        self.In_Cols_Num = []

    def generate_initial_patterns(self):
        # initial routes pool
        impactSol = initializePathsWithImpact(self.d, self.n, self.a, self.b, self.q, self.Q)
        initial_routes = impactSol[:]
        self.routes = deepcopy(initial_routes)  # assign initial routes pool to routes element in VRP class

        A = np.zeros((self.n, len(self.routes)))
        c = np.zeros(len(self.routes))
        addRoutesToMaster(self.routes, A, c, self.d)  # update A and c with routes

        self.A = deepcopy(A)
        self.c = deepcopy(c)

    def update_col_con_number(self):
        A = deepcopy(self.A)
        self.In_Cons_Num = np.count_nonzero(A, axis=0)  # count the number of customer in each route
        self.In_Cols_Num = np.count_nonzero(A, axis=1)  # count the number of routes that each customer is in
        # print("In function update_col_con_number() in class VRP")
        # print("Update the count the number of customer in each route: ", self.In_Cons_Num)
        # print("Update the count the number of routes that each customer is in: ", self.In_Cols_Num)

    def createMasterProblem(self):  # same as createMasterProblem(A, costs, n, vehicleNumber)
        A = self.A
        costs = self.c
        n = self.n
        vehicleNumber = self.Kdim

        model = gp.Model("Master problem")
        model.Params.OutputFlag = 0
        y = model.addMVar(shape=A.shape[1], vtype=gp.GRB.CONTINUOUS, name="y")
        model.setObjective(costs @ y, gp.GRB.MINIMIZE)
        model.addConstr(A @ y == np.ones(A.shape[0]))
        model.write("MasterModel.lp")
        model.Params.LogToConsole = 0

        return model

    def solve_subproblem_return_actions(self, rc, test=False):
        n = self.n  # number of customers
        q = self.q  # demand of customers
        d = self.d  # distance matrix
        readyt = self.a  # earliest time to serve
        duedate = self.b  # latest time to serve
        Q = self.Q  # capacity of vehicles

        M = gp.GRB.INFINITY  # 1e+100
        # Time windows reduction
        a, b = reduceTimeWindows(n, d, readyt, duedate)
        # Reduce max capacity to boost algorithm
        if sum(q) < Q:
            Q = sum(q)
        T = max(b)

        # Init necessary data structure
        f = list()  # paths cost data struct
        p = list()  # paths predecessor data struct
        f_tk = list()  # cost of the best path that does not pass for
        # predecessor (we'll call it alternative path)
        paths = []
        paths_tk = []
        for j in range(n + 2):
            paths.append([])
            paths_tk.append([])
            for qt in range(Q - q[j]):
                paths[-1].append([])
                paths_tk[-1].append([])
                for tm in range(b[j] - a[j]):
                    paths[-1][-1].append([])
                    paths_tk[-1][-1].append([])
            mat = np.zeros((Q - q[j], b[j] - a[j]))
            p.append(mat - 1)
            f.append(mat + M)
            f_tk.append(mat + M)
        f[0][0, 0] = 0
        f_tk[0][0, 0] = 0
        L = set()  # Node to explore
        L.add(0)

        # Algorithm
        computation_counter = 0
        while L:
            if test:  ## we will test on large instances, so set up some computation limits
                if computation_counter >= 20:
                    computation_counter += 1
                    break
            else:
                computation_counter += 1  ## but for training we want to train without such limits as instances are small

            i = L.pop()
            if i == n + 1:
                continue

            # Explore all possible arcs (i,j)
            for j in range(1, n + 2):
                if i == j:
                    continue
                for q_tk in range(q[i], Q - q[j]):
                    for t_tk in range(a[i], b[i]):
                        if p[i][q_tk - q[i], t_tk - a[i]] != j:
                            if f[i][q_tk - q[i], t_tk - a[i]] < M:
                                for t in range(max([a[j], int(t_tk + d[i, j])]), b[j]):
                                    if f[j][q_tk, t - a[j]] > f[i][q_tk - q[i], t_tk - a[i]] + rc[i, j]:
                                        # if the current best path is suitable to
                                        # become the alternative path
                                        if p[j][q_tk, t - a[j]] != i \
                                                and p[j][q_tk, t - a[j]] != -1 \
                                                and f[j][q_tk, t - a[j]] < M \
                                                and f[j][q_tk, t - a[j]] < f_tk[j][q_tk, t - a[j]]:
                                            f_tk[j][q_tk, t - a[j]] = f[j][q_tk, t - a[j]]
                                            paths_tk[j][q_tk][t - a[j]] = \
                                                paths[j][q_tk][t - a[j]][:]
                                        # update f
                                        f[j][q_tk, t - a[j]] = \
                                            f[i][q_tk - q[i], t_tk - a[i]] + rc[i, j]
                                        # update path that leads to node j
                                        paths[j][q_tk][t - a[j]] = \
                                            paths[i][q_tk - q[i]][t_tk - a[i]] + [j]
                                        # Update predecessor
                                        p[j][q_tk, t - a[j]] = i
                                        L.add(j)
                                    # if the path is suitable to be the alternative
                                    elif p[j][q_tk, t - a[j]] != i \
                                            and p[j][q_tk, t - a[j]] != -1 \
                                            and f_tk[j][q_tk, t - a[j]] > \
                                            f[i][q_tk - q[i], t_tk - a[i]] + rc[i, j]:
                                        f_tk[j][q_tk, t - a[j]] = \
                                            f[i][q_tk - q[i], t_tk - a[i]] + rc[i, j]
                                        paths_tk[j][q_tk][t - a[j]] = \
                                            paths[i][q_tk - q[i]][t_tk - a[i]] + [j]
                        else:  # if predecessor of i is j
                            if f_tk[i][q_tk - q[i], t_tk - a[i]] < M:
                                for t in range(max([a[j], int(t_tk + d[i, j])]), b[j]):
                                    if f[j][q_tk, t - a[j]] > \
                                            f_tk[i][q_tk - q[i], t_tk - a[i]] + rc[i, j]:
                                        # if the current best path is suitable to
                                        # become the alternative path
                                        if p[j][q_tk, t - a[j]] != i \
                                                and p[j][q_tk, t - a[j]] != -1 \
                                                and f[j][q_tk, t - a[j]] < M \
                                                and f[j][q_tk, t - a[j]] < \
                                                f_tk[j][q_tk, t - a[j]]:
                                            f_tk[j][q_tk, t - a[j]] = f[j][q_tk, t - a[j]]
                                            paths_tk[j][q_tk][t - a[j]] = \
                                                paths[j][q_tk][t - a[j]][:]
                                        # update f, path and bucket
                                        f[j][q_tk, t - a[j]] = \
                                            f_tk[i][q_tk - q[i], t_tk - a[i]] + rc[i, j]
                                        paths[j][q_tk][t - a[j]] = \
                                            paths_tk[i][q_tk - q[i]][t_tk - a[i]] + [j]
                                        p[j][q_tk, t - a[j]] = i
                                        L.add(j)
                                    # if the alternative path of i is suitable to
                                    # be the alternate of j
                                    elif p[j][q_tk, t - a[j]] != i \
                                            and p[j][q_tk, t - a[j]] != -1 \
                                            and f_tk[j][q_tk, t - a[j]] > \
                                            f_tk[i][q_tk - q[i], t_tk - a[i]] + rc[i, j]:
                                        f_tk[j][q_tk, t - a[j]] = \
                                            f_tk[i][q_tk - q[i], t_tk - a[i]] + rc[i, j]
                                        paths_tk[j][q_tk][t - a[j]] = \
                                            paths_tk[i][q_tk - q[i]][t_tk - a[i]] + [j]

        # Return all the routes with negative cost
        routes = list()
        rcosts = list()
        qBest, tBest = np.where(f[n + 1] < -1e-9)

        for i in range(len(qBest)):
            newRoute = [0] + paths[n + 1][qBest[i]][tBest[i]]
            if newRoute not in routes:
                routes.append(newRoute)
                rcosts.append(f[n + 1][qBest[i]][tBest[i]])

        costs = np.zeros(len(routes))
        for i in range(len(routes)):
            cost = d[routes[i][0], routes[i][1]]
            for j in range(1, len(routes[i]) - 1):
                cost += d[routes[i][j], routes[i][j + 1]]
            costs[i] = cost

        costs = list(costs)
        # print("new routes",routes)

        return routes, rcosts, costs  ## return all available actions (new routes generated, called routes here) and their rc

    def initialize(self, test_or_not=False):
        self.generate_initial_patterns()  # initialize the route pooling
        self.total_steps = 0
        routes = self.routes

        # print("=====================================")
        # print("current route: ", routes)

        self.update_col_con_number()  # update the features

        master_problem = self.createMasterProblem()  # create the master problem
        master_problem.optimize()  # solve the master problem

        # # Compute reduced costs
        # constr = masterModel.getConstrs()
        # pi_i = [0.] + [const.pi for const in constr] + [0.]
        # for i in range(n+2):
        #     for j in range(n+2):
        #         rc[i,j] = d[i,j] - pi_i[i]
        # if not np.where(rc < -1e-9):
        #     break
        # self.RC = np.zeros(len(self.routes))

        self.ColumnSol_Val = np.asarray(master_problem.x)  # get the column solution ([0,1])
        # print("ColumnSol_Val", self.ColumnSol_Val)
        self.ColumnIs_Basic = np.asarray(master_problem.vbasis) + np.ones(len(routes))  # get the basic of columns (0/1)
        # print("ColumnIs_Basic", self.ColumnIs_Basic)
        self.objVal_history.append(master_problem.objVal)  # store the objective value
        # print("objVal = ", master_problem.objVal)

        dual_variables = [constraint.pi for constraint in master_problem.getConstrs()]
        self.Shadow_Price = dual_variables
        # print("Shadow_Price", self.Shadow_Price)

        # Compute reduced costs
        constr = master_problem.getConstrs()
        pi_i = [0.] + [const.pi for const in constr] + [0.]
        rc = np.zeros((self.n + 2, self.n + 2))
        for i in range(self.n + 2):
            for j in range(self.n + 2):
                rc[i, j] = self.d[i, j] - pi_i[i]
        # print("reduced cost",rc)

        toolong = False
        time_before = time.time()
        columns_to_select, reduced_costs, route_costs = self.solve_subproblem_return_actions(rc, test_or_not)
        time_after = time.time()

        # print("routes: ", columns_to_select)
        # print("reduced costs: ", reduced_costs)
        # print("route costs: ", route_costs)

        if not test_or_not:  # if the instance takes too long to solve during training, skip it
            if time_after - time_before >= 100.0:
                toolong = True
        else:  # if it is testing, do not skip any instances
            toolong = False  # test every instances

        self.available_action = (columns_to_select, [reduced_costs, route_costs])

        self.stay_in = list(np.zeros(len(routes)))  # initialize the stay_in and stay_out
        self.stay_out = list(np.zeros(len(routes)))
        self.just_left = list(np.zeros(len(routes)))  # initialize the just_left and just_enter
        self.just_enter = list(np.zeros(len(routes)))

        Tmp_Reward = 0
        isDone = False

        return Tmp_Reward, isDone, toolong

    # action is a route
    def step(self, action, test_or_not=False):
        tmp_reward = 0
        self.total_steps += 1
        isDone = False

        ## historical info
        last_columns_to_select, columns_info = deepcopy(self.available_action)  # (column) (r, cost)
        last_basis = deepcopy(self.ColumnIs_Basic[:])
        last_basis = np.append(last_basis, 0)

        self.routes.append(action)  # add the new route to the route pool
        idx = 0  # find the index of the action in the last_columns_to_select
        for tmp_col in last_columns_to_select:  # if action is the route that is in last_columns_to_select
            if tmp_col == action:
                break
            idx += 1
        # if idx == len(last_columns_to_select), it means that this action is in last_columns_to_select

        routes = deepcopy(self.routes)

        ## just append one cost
        self.c = np.append(self.c, columns_info[1][idx])  # update cost
        # self.rc = np.append(self.rc, columns_info[0][idx])  # update reduced cost

        add_A = np.zeros((self.n, 1))  # a 0's vector of number of customer, col vector, can be added at the right of A

        for j in range(1, len(action) - 1):  # need to travel all the action to update add_A
            # the first and the last one is depot
            add_A[action[j] - 1] += 1

        # print(action)
        # print(add_A)
        # exit(0)

        self.A = np.c_[self.A, add_A]  # same rows, add a colomn

        self.update_col_con_number()  # since new column is added, update the features

        master_problem = self.createMasterProblem()
        master_problem.optimize()

        dual_variables = [constraint.pi for constraint in master_problem.getConstrs()]
        self.Shadow_Price = dual_variables
        self.ColumnSol_Val = np.asarray(master_problem.x)
        self.ColumnIs_Basic = np.asarray(master_problem.vbasis) + np.ones(len(routes))
        # you can either stay in the basis, leave the basis, or enter the basis
        difference = last_basis - self.ColumnIs_Basic

        # update the dynamic basis info based on difference
        self.just_left = list(np.zeros(len(difference) - 1))
        self.just_enter = list(np.zeros(len(difference) - 1))
        for i in range(len(difference) - 1):
            if difference[i] == 1:
                self.just_left[i] = 1
                self.stay_in[i] = 0
            elif difference[i] == -1:
                self.just_enter[i] = 1
                self.stay_out[i] = 0
            elif difference[i] == 0:
                if last_basis[i] == 1:
                    self.stay_in[i] += 1
                else:
                    self.stay_out[i] += 1

        # append info for the new node; for just enter, look at column is basic
        self.just_left.append(0)
        self.stay_out.append(0)
        self.stay_in.append(0)

        if self.ColumnIs_Basic[-1] == 1:  # if new route is basis, then add 1 to just_enter
            self.just_enter.append(1)
        else:  # else, add 0 to just_enter
            self.just_enter.append(0)

        self.objVal_history.append(master_problem.objVal)

        rc = np.zeros((self.n + 2, self.n + 2))
        constr = master_problem.getConstrs()
        pi_i = [0.] + [const.pi for const in constr] + [0.]
        for i in range(self.n + 2):
            for j in range(self.n + 2):
                rc[i, j] = self.d[i, j] - pi_i[i]
        # print("reduced cost",rc)

        new_routes, action_rcosts, action_costs = self.solve_subproblem_return_actions(rc, test_or_not)
        if not new_routes:  # if there is no new routes can be generated, then the problem is solved in this episode
            isDone = True
            tmp_reward = 100 * (self.objVal_history[-2] - self.objVal_history[-1]) / self.objVal_history[0]
        else:
            self.available_action = (new_routes, [action_rcosts, action_costs])
        tmp_reward -= 1

        return tmp_reward, isDone


# s0_augmented, a0, r, is_done, s1_augmented, total
class Transition(object):
    def __init__(self, s0_augmented, a0, r: float, is_done: bool, s1_augmented, action_info_0, total_0, total_1):
        self.data = (s0_augmented, a0, r, is_done, s1_augmented, action_info_0, total_0, total_1)

    @property
    def s0(self):
        return self.data[0]

    @property
    def a0(self):
        return self.data[1]

    @property
    def reward(self):
        return self.data[2]

    @property
    def is_done(self):
        return self.data[3]

    @property
    def s1(self):
        return self.data[4]

    @property
    def action_info_0(self):
        return self.data[5]

    @property
    def total_0(self):
        return self.data[6]

    @property
    def total_1(self):
        return self.data[7]

    def __iter__(self):  # iteration
        return iter(self.data)

    def __str__(self):  # print string
        edge_num1 = len(self.data[0][1][1])  # s0_augmented
        edge_num2 = len(self.data[-1][1][1])  # total 1
        result = 'transit from bipartite graph with edge' + str(edge_num1) + ' to ' + str(
            edge_num2) + ' collecting the reward ' + str(self.data[2])
        # return 'transit from bipartite graph ' + col_string1 + ' connects to ' + con_string1 + ' to ' + col_string2 + ' connects to ' + con_string2 +' collecting the reward ' + str(self.data[2])
        return result


class Episode(object):
    def __init__(self, e_id: int = 0) -> None:
        self.total_reward = 0
        self.trans_list = []
        self.name = str(e_id)

    def push(self, trans: Transition) -> float:
        self.trans_list.append(trans)
        self.total_reward += trans.reward
        return self.total_reward

    @property
    def len(self):
        return len(self.trans_list)

    def __str__(self):
        return "episode {0:<4} {1:>4} steps,total reward:{2:<8.2f}". \
            format(self.name, self.len, self.total_reward)

    def print_detail(self):
        print("detail of ({0}):".format(self))
        for i, trans in enumerate(self.trans_list):
            print("step{0:<4} ".format(i), end=" ")
            print(trans)

    def is_complete(self) -> bool:  # check if an episode is an complete episode
        if self.len == 0:
            return False
        return self.trans_list[self.len - 1].is_done

    def sample(self, batch_size=1):  # random generate a trans
        return random.sample(self.trans_list, k=batch_size)

    def __len__(self) -> int:
        return self.len


class Experience(object):
    def __init__(self, capacity: int = 200000):
        self.capacity = capacity
        self.episodes = []
        self.next_id = 0
        self.total_trans = 0

    def __str__(self):
        return "exp info:{0:5} episodes, memory usage {1}/{2}". \
            format(self.len, self.total_trans, self.capacity)

    def __len__(self):
        return self.len

    @property
    def len(self):
        return len(self.episodes)

    def _remove(self, index=0):  # remove an episode, default the first one.
        # the index of the episode to remove
        # return: if exists return the episode else return None
        if index > self.len - 1:
            raise (Exception("invalid index"))
        if self.len > 0:
            episode = self.episodes[index]
            self.episodes.remove(episode)
            self.total_trans -= episode.len
            return episode
        else:
            return None

    def push(self, trans):
        if self.capacity <= 0:
            return
        while self.total_trans >= self.capacity:
            episode = self._remove(0)

        cur_episode = None
        if self.len == 0 or self.episodes[self.len - 1].is_complete():
            cur_episode = Episode(self.next_id)
            self.next_id += 1
            self.episodes.append(cur_episode)
        else:
            cur_episode = self.episodes[self.len - 1]
        self.total_trans += 1
        return cur_episode.push(trans)  # return total reward of an episode

    def sample(self, batch_size=32):  # randomly sample some transitions from agent's experience.abs
        # input: number of transitions need to be sampled
        # output: list of Transition.
        sample_trans = np.asarray([])
        for _ in range(batch_size):
            index = int(random.random() * self.len)
            sample_trans = np.append(sample_trans, self.episodes[index].sample())
        return sample_trans

    def sample_episode(self, episode_num=1):  # sample episode
        return random.sample(self.episodes, k=episode_num)

    @property
    def last_episode(self):
        if self.len > 0:
            return self.episodes[self.len - 1]
        return None


class Agent(object):
    def __init__(self, initial_env, capacity, hidden_dim, batch_size, epochs, embedding_size, cons_num_features,
                 vars_num_features, learning_rate, seed_):  # env, A, S (feature, a_info)
        self.env = initial_env
        # add the env and available action will be added in the learning_method
        # self.A = self.env.available_action

        self.AAction = []
        self.experience = Experience(capacity=capacity)

        # S record the current super state for the agent
        # self.S = self.get_aug_state()
        self.S = []

        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.cons_num_features = cons_num_features
        self.vars_num_features = vars_num_features
        self.lr = learning_rate
        self.seed = seed_
        self.batch_size = batch_size
        self.epochs = epochs
        self.behavior_Q = BipartiteGNN(embedding_size=self.embedding_size, cons_num_features=self.cons_num_features,
                                       vars_num_features=self.vars_num_features, learning_rate=self.lr, seed=self.seed)
        self.target_Q = BipartiteGNN(embedding_size=self.embedding_size, cons_num_features=self.cons_num_features,
                                     vars_num_features=self.vars_num_features, learning_rate=self.lr, seed=self.seed)
        self._update_target_Q()

    def _update_target_Q(self):
        self.target_Q.set_weights(deepcopy(self.behavior_Q.variables))

    def get_max(self, total_1, s):
        Q_s = self.target_Q.callGNN(s)
        Q_s_for_action = Q_s[-total_1::]
        return np.max(Q_s_for_action)

    # get augmented state from the current environment
    def get_aug_state(self):
        actions, action_info = deepcopy(self.env.available_action)  # col, (rc, cost))
        reduced_costs = action_info[0]

        total_added = len(actions)  # number of columns in this actions
        patterns = self.env.routes[:]
        col_num = len(patterns)  # number of total columns
        is_action = np.asarray([0] * col_num)  # the number of columns * 0, original one
        patterns.extend(actions)  # add the new columns to the old route pool

        cons_num = self.env.n  # number of customer
        column_features = []
        cons_features = []
        edge_indices = [[], []]

        MatA = deepcopy(self.env.A)  # the constraint matrix
        cost_c = deepcopy(self.env.c)  # the cost of each column

        newMat = np.zeros((self.env.n, len(actions)))  # the constraint matrix for the new columns
        newCosts = np.zeros(len(actions))  # the cost vector of the new columns
        addRoutesToMaster(actions, newMat, newCosts, self.env.d)  # update new A and c

        MatA = np.c_[MatA, newMat]  # update A **

        # update the number of columns in each constraint
        In_Cons_Num = np.count_nonzero(MatA, axis=0)

        # ColumnSol_Val = self.env.ColumnSol_Val[:]
        ColumnSol_Val = np.append(self.env.ColumnSol_Val[:],
                                  np.zeros(total_added))  # extend col_sol, the new route value is 0
        cost_c = np.append(cost_c, newCosts)  # update c **

        # update the stay_in, stay_out, just_left, just_enter, all the values of new routes are 0
        stay_in = self.env.stay_in[:]
        stay_in = np.append(stay_in, np.zeros(total_added))
        stay_out = self.env.stay_out[:]
        stay_out = np.append(stay_out, np.zeros(total_added))
        just_left = self.env.just_left[:]
        just_left = np.append(just_left, np.zeros(total_added))
        just_enter = self.env.just_enter[:]
        just_enter = np.append(just_enter, np.zeros(total_added))

        # update the is_action, the new routes' value are all 1
        is_action = np.append(is_action, np.ones(total_added))  # make sure that we only add the new route

        Shadow_Price = self.env.Shadow_Price[:]
        In_Cols_Num = np.count_nonzero(MatA, axis=1)

        Shadow_Price = np.asarray(Shadow_Price).reshape(-1, 1)  # column vector
        In_Cons_Num = np.asarray(In_Cons_Num).reshape(-1, 1)
        In_Cols_Num = np.asarray(In_Cols_Num).reshape(-1, 1)
        ColumnSol_Val = np.asarray(ColumnSol_Val).reshape(-1, 1)
        cost_c = np.asarray(cost_c).reshape(-1, 1)
        stay_in = np.asarray(stay_in).reshape(-1, 1)
        stay_out = np.asarray(stay_out).reshape(-1, 1)

        # normalization
        Scaler_SP = MinMaxScaler()
        Shadow_Price = Scaler_SP.fit_transform(Shadow_Price)

        Scaler_IConsN = MinMaxScaler()
        In_Cons_Num = Scaler_IConsN.fit_transform(In_Cons_Num)

        Scaler_IColsN = MinMaxScaler()
        In_Cols_Num = Scaler_IColsN.fit_transform(In_Cols_Num)

        Scaler_CSV = MinMaxScaler()
        ColumnSol_Val = Scaler_CSV.fit_transform(ColumnSol_Val)

        Scaler_W = MinMaxScaler()
        cost_c = Scaler_W.fit_transform(cost_c)

        Scaler_si = MinMaxScaler()
        stay_in = Scaler_si.fit_transform(stay_in)

        Scaler_out = MinMaxScaler()
        stay_out = Scaler_out.fit_transform(stay_out)

        Shadow_Price = list(Shadow_Price.T[0])
        In_Cons_Num = list(In_Cons_Num.T[0])
        In_Cols_Num = list(In_Cols_Num.T[0])
        ColumnSol_Val = list(ColumnSol_Val.T[0])
        cost_c = list(cost_c.T[0])
        stay_in = list(stay_in.T[0])
        stay_out = list(stay_out.T[0])

        # constraint nodes
        for j in range(cons_num):  # cons_num = customer number
            con_feat = [Shadow_Price[j], In_Cols_Num[j]]
            cons_features.append(con_feat)

        # col nodes
        for i in range(col_num):
            col_feat = [In_Cons_Num[i], ColumnSol_Val[i], cost_c[i], stay_in[i], stay_out[i], just_left[i],
                        just_enter[i], is_action[i]]
            column_features.append(col_feat)

        ## get edges going
        for m in range(len(MatA[0])):
            for n in range(len(MatA)):
                if MatA[n][m] != 0:
                    # then mth column is connected to nth cons
                    edge_indices[0].append(m)  # restore the column index
                    edge_indices[1].append(n)  # restore the cons index

        edge_indices = np.asarray(edge_indices)
        edge_indices[[0, 1]] = edge_indices[[1, 0]]  # cons index, col index

        cons_features = np.asarray(cons_features)
        column_features = np.asarray(column_features)

        ## need this total_added for reading the Q values, need actions to select onne pattern after read Q values
        aug_state, action_info = ((cons_features, edge_indices, column_features), (total_added, actions))
        return aug_state, action_info

    def act(self, a0):
        s0_augmented, action_info_0 = self.S  # record the current super state for the agent
        total_0 = deepcopy(action_info_0[0])  # the number of added columns in the 'current' super state

        # step change the environment, update all the information used for agent to construct state
        temp_r, isDone = self.env.step(a0)  # take action a0, get reward and is_done

        s1_augmented, action_info_1 = self.get_aug_state()  # get the next super state for the agent
        total_1 = action_info_1[0]  # the number of added columns in the 'next' super state
        trans = Transition(s0_augmented, a0, temp_r, isDone, s1_augmented, action_info_0, total_0, total_1)
        total_reward = self.experience.push(trans)
        self.S = s1_augmented, action_info_1

        return s1_augmented, temp_r, isDone, total_reward

    # action information: (# of added columns, cols), s: super state (col feature, edges, cons feature)
    def policy(self, action_info_input, s_input, epsilon=None):
        # print("DQNAgent policy method is called")
        total_added, Actions = action_info_input
        Q_s = self.behavior_Q.callGNN(s_input)  # get logit vector for all actions
        Q_s_for_action = Q_s[-total_added::]

        rand_value = np.random.random()
        if epsilon is not None and rand_value < epsilon:
            return random.choice(list(Actions))
        else:
            idx = int(np.argmax(Q_s_for_action))
            return Actions[idx]

    def _learn_from_memory(self, gamma, learning_rate):
        ## trans_pieces is a list of transitions
        trans_pieces = self.sample(self.batch_size)  # Get transition data
        states_0 = np.vstack([x.s0 for x in trans_pieces])  # as s0 is a list, so vstack
        actions_0 = np.array([x.a0 for x in trans_pieces])
        reward_1 = np.array([x.reward for x in trans_pieces])
        is_dones = np.array([x.is_done for x in trans_pieces])
        states_1 = np.vstack([x.s1 for x in trans_pieces])
        action_info_tmp = np.vstack([x.action_info_0 for x in trans_pieces])
        totals_0 = np.vstack([x.total_0 for x in trans_pieces])
        totals_1 = np.vstack([x.total_1 for x in trans_pieces])

        y_batch = []
        for i in range(len(states_0)):
            ### get the index of action that is taken at s0
            acts_0 = action_info_tmp[i][1]
            act_0 = list(actions_0[i])

            print("act:", act_0)
            print("acts:", acts_0, '\n')

            idx = 0  # how many actions are taken before the action at s0
            for act in acts_0:
                if act == act_0:
                    break
                idx += 1

            y = self.target_Q.callGNN(states_0[i]).numpy()
            
            print(y, '\n', y.shape)
            print(totals_0[i][0])
            print(y[0:-totals_0[i][0]])
            # exit()
            #### set the non action terms to be 0
            y[0:-totals_0[i][0]] = 0

            if is_dones[i]:
                Q_target = reward_1[i]
            else:
                ### the number of actions for state 1 is used to get Q_target
                Q_max = self.get_max(totals_1[i][0], states_1[i])
                Q_target = reward_1[i] + gamma * Q_max

            y[-totals_0[i][0] + idx] = Q_target

            y_batch.append(np.asarray(y))

        y_batch = np.asarray(y_batch)
        X_batch = states_0

        # still train behavior_Q
        loss = self.behavior_Q.train_or_test(X_batch, y_batch, totals_0, actions_0, action_info_tmp, True)
        print("In function _learn_from_memory() the loss is,", loss)
        self._update_target_Q()

        return loss

    def learning_method(self, instance, gamma, learning_rate, epsilon, display=False):
        epochs = self.epochs
        self.env = instance
        self.S = self.get_aug_state()

        time_in_episode, total_reward = 0, 0
        isDone = False
        loss = 0
        while not isDone:
            s0_aug = self.S[0]
            temp_action_info = self.S[1]

            # a0 is selected based on behavior_Q
            if len(temp_action_info[1]) == 0:
                ## if no available actions,end this episode
                break

            a0 = self.policy(temp_action_info, s0_aug, epsilon)
            print("Added route is {} out of {}".format(a0, len(temp_action_info[1])))

            s1_augmented, r, isDone, total_reward = self.act(a0)

            if self.total_trans > self.batch_size:
                for e in range(epochs):
                    loss += self._learn_from_memory(gamma, learning_rate)
                # loss/=epochs
            # s0 = s1
            time_in_episode += 1

        loss /= (time_in_episode * epochs)
        print("The loss is,", loss)
        if display:
            print("epsilon:{:3.2f},loss:{:3.2f},{}".format(epsilon, loss, self.experience.last_episode))
        return time_in_episode, total_reward

    # In the early stages of training, we want to explore with a larger epsilon value,
    # so a larger initial value needs to be set. However, in the later stages of training,
    # we want the agent to gradually reduce its exploration frequency,
    # so we need to limit the epsilon value to a smaller range.
    @staticmethod
    def _decayed_epsilon(cur_episode: int, min_epsilon: float, max_epsilon: float, target_episode: int) -> float:
        slope = (min_epsilon - max_epsilon) / target_episode
        intercept = max_epsilon
        return max(min_epsilon, slope * cur_episode + intercept)

    def learning(self, epsilon=0.05, decaying_epsilon=True, gamma=0.9,
                 learning_rate=3e-4, max_episode_num=153, display=False, min_epsilon=1e-2, min_epsilon_ratio=0.8,
                 model_index=0):
        total_time, episode_reward, num_episode = 0, 0, 0
        total_time_set, episode_reward_set, num_episode_set, history = [], [], [], []
        schedule = np.load("schedule.npy")

        for i in range(max_episode_num):
            if epsilon is None:
                epsilon = 1e-10
            elif decaying_epsilon:
                target_e = int(max_episode_num * min_epsilon_ratio)
                epsilon = self._decayed_epsilon(num_episode + 1, min_epsilon, epsilon, target_e)

            n, problem_name = schedule[i]
            # print("in Agent class ", len(schedule))

            VRP_instance = VRP(problem_name, int(n))

            too_long = VRP_instance.initialize()[2]

            # do while loop until is_done = True
            time_in_episode, episode_reward = self.learning_method(VRP_instance, gamma, learning_rate, epsilon)

            num_episode += 1

            total_time_set.append(time_in_episode)
            episode_reward_set.append(episode_reward)
            num_episode_set.append(num_episode)
            history.append(VRP_instance.objVal_history[-1])

            print("Episode: " + str(i) + " takes " + str(time_in_episode) + " steps with obj " + str(
                VRP_instance.objVal_history[-1]))

        model_save_name_temp = 'Model.pt'
        path_model_temp = 'save/' + model_save_name_temp
        self.target_Q.save_state(path_model_temp)

        return total_time_set, episode_reward_set, num_episode_set, history

    def sample(self, batch_size=32):
        return self.experience.sample(batch_size)

    @property
    def total_trans(self):
        return self.experience.total_trans

    def last_episode_detail(self):
        self.experience.last_episode.print_detail()


def follow_policy(DQNAgent, action_info_input, s):  # DQN selects an action
    total_added, Actions = action_info_input
    Q_s = DQNAgent.target_Q.callGNN(s)
    Q_s_for_action = Q_s[-total_added::]
    # rand_value = np.random.random()
    idx = int(np.argmax(Q_s_for_action))
    return Actions[idx]


class PARAMETERS(object):
    def __init__(self):
        self.seed = 5

        ## parameters about neural network
        self.lr = 1e-3  ##
        self.batch_size = 16
        self.hidden_dim = 32
        self.epochs = 5
        self.embedding_size = 32
        self.cons_num_features = 2
        self.vars_num_features = 8

        ## parameters of RL algorithm
        self.gamma = 0.99  ##
        self.epsilon = 0.2
        self.min_epsilon = 0.2
        self.min_epsilon_ratio = 0.99
        self.decaying_epsilon = False
        self.step_penalty = 1
        self.alpha_obj_weight = 5  ##
        self.action_pool_size = 10
        self.max_episode_num = 4
        self.capacity = 20000

        self.model_index = 3  #####


# data assignment
Parameters = PARAMETERS()

random.seed(Parameters.seed)
np.random.seed(Parameters.seed)

epsilon_ = Parameters.epsilon
decaying_epsilon_ = Parameters.decaying_epsilon
gamma_ = Parameters.gamma
alpha_ = Parameters.alpha_obj_weight
max_episode_num_ = Parameters.max_episode_num  # Parameters.max_episode_num
min_epsilon_ = Parameters.min_epsilon
min_epsilon_ratio_ = Parameters.min_epsilon_ratio
capacity_ = Parameters.capacity
hidden_dim_ = Parameters.hidden_dim
batch_size_ = Parameters.batch_size
epochs_ = Parameters.epochs
embedding_size_ = Parameters.embedding_size
cons_num_features_ = Parameters.cons_num_features
vars_num_features_ = Parameters.vars_num_features
learning_rate_ = Parameters.lr
model_index_ = Parameters.model_index
seed_fix = Parameters.seed

display_ = False

print("************************************* train ************************************************")

t_train_start = time.time()

DQN = Agent(initial_env=None, capacity=capacity_, hidden_dim=hidden_dim_, batch_size=batch_size_, epochs=epochs_,
            embedding_size=embedding_size_, cons_num_features=cons_num_features_,
            vars_num_features=vars_num_features_, learning_rate=learning_rate_,
            seed_=seed_fix)

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

total_times, episode_rewards, num_episodes, temp_his = DQN.learning(epsilon=epsilon_,
                                                                    decaying_epsilon=decaying_epsilon_,
                                                                    gamma=gamma_, learning_rate=learning_rate_,
                                                                    max_episode_num=max_episode_num_, display=display_,
                                                                    min_epsilon=min_epsilon_,
                                                                    min_epsilon_ratio=min_epsilon_ratio_,
                                                                    model_index=model_index_)
t_train_end = time.time()
t_train = t_train_end - t_train_start

print("Training time is: ", t_train)

print("************************************* test ************************************************")
VRP_instance2 = VRP("c101.txt", 20)
too_long = VRP_instance2.initialize()[2]
#
# print(DQN.S[0][0])
# print(len(DQN.S[0][0]), "\n")
#
# print(DQN.S[0][1])
# print(len(DQN.S[0][1]), " ", len(DQN.S[0][1][0]), "\n")
#
# print(DQN.S[0][2])
# print(len(DQN.S[0][2]), "\n")
#
# print(DQN.S[1][0], "\n")
#
# print(DQN.S[1][1])
# print(len(DQN.S[1][1]), "\n")
# exit(0)

model_save_name = 'Model.pt'
path_model = 'save/' + model_save_name

DQN.target_Q.restore_state(path_model)
DQN.behavior_Q.restore_state(path_model)

t_test_gready_start = time.time()
is_done = False
while True:
    if is_done:
        break

    action = VRP_instance2.available_action[0][-1]
    reward, is_done = VRP_instance2.step(action)
    print("Added route is {} out of {}".format(action, len(VRP_instance2.available_action[0])))
t_test_gready_end = time.time()

history_opt_g = VRP_instance2.objVal_history

obj_greedy = history_opt_g[-1]
steps_g = len(history_opt_g)
print("Greedy takes {} steps to reach obj {} with time {}".format(steps_g, obj_greedy,
                                                                  t_test_gready_end - t_test_gready_start))

DQN.env = VRP_instance2
DQN.S = DQN.get_aug_state()
is_done = False
t_test_start = time.time()
while True:
    if is_done:
        break

    action_info = DQN.S[1]
    s = DQN.S[0]
    action = follow_policy(DQN, action_info, s)
    reward, is_done = VRP_instance2.step(action)
    DQN.S = DQN.get_aug_state()

    print("Added route is {} out of {}".format(action, len(action_info[1])))
t_test_end = time.time()
test_time = t_test_end - t_test_start

history_opt_rl = VRP_instance2.objVal_history

obj_RL = history_opt_rl[-1]
steps_RL = len(history_opt_rl)

print("RL takes {} steps to reach obj {} with time {}".format(steps_RL, obj_RL, t_test_end - t_test_start))
