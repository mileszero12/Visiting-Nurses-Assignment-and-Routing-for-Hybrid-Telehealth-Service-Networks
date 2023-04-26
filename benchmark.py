import sys
from gurobipy import *
#import numpy
import random
import math
import csv
import time
#import pandas as pd
#import matplotlib.pyplot as plt


#model=Model()
#model=read("0.mps")
#model.optimize()





def gurobiSolve(name1, name2, name, social, ratio):
    with open(name1 + name2 + name + '.txt') as f:
        numc = int(next(f).split()[0])
        nump1 = int(next(f).split()[0])
        nump2 = int(next(f).split()[0])
        numvr = int(next(f).split()[0])
        numnr = int(next(f).split()[0])
        numhr = int(next(f).split()[0])
        nump = nump1 + nump2

        data = []
        for line in f:
            data.append([float(x) for x in line.split()])

    txt3 = []
    xc = []
    yc = []
    fee = []
    for c in range(numc):
        txt3.append(int(data[c][0]))
        xc.append(data[c][1])
        yc.append(data[c][2])
        fee.append(data[c][3])

    salary_vr = []
    for vr in range(numvr):
        salary_vr.append(data[numc + vr][2])

    salary_nr = []
    for nr in range(numnr):
        salary_nr.append(data[numc + numvr + nr][2])

    salary_hr = []
    for hr in range(numhr):
        salary_hr.append(data[numc + numvr + numnr + hr][2])

    cur_index = numc + numvr + numnr + numhr

    txt1 = []
    p1_x = []
    p1_y = []
    p1_st = []
    p1_ser = []
    p1_ed = []

    for p1 in range(nump1):
        txt1.append(int(data[cur_index + p1][0]))
        p1_x.append(data[cur_index + p1][2])
        p1_y.append(data[cur_index + p1][3])
        p1_ser.append(data[cur_index + p1][4])
        p1_st.append(data[cur_index + p1][5])
        p1_ed.append(data[cur_index + p1][6])

    txt2 = []
    p2_x = []
    p2_y = []
    p2_st = []
    p2_ser = []
    p2_ed = []

    cur_index += nump1;
    for p1 in range(nump2):
        txt2.append(int(data[cur_index + p1][0]))
        p2_x.append(data[cur_index + p1][2])
        p2_y.append(data[cur_index + p1][3])
        p2_ser.append(data[cur_index + p1][4])
        p2_st.append(data[cur_index + p1][5])
        p2_ed.append(data[cur_index + p1][6])

    px = p1_x + p2_x
    py = p1_y + p2_y
    st = p1_st + p2_st
    ed = p1_ed + p2_ed
    ser = p1_ser + p2_ser

    pcx = px + xc
    pcy = py + yc

    numc1 = numc - 1

    #plot(p1_x, p1_y, txt1, p2_x, p2_y, txt2, xc, yc, txt3, name1, name)

    def travel(j1, j2):
        return (((px[j1] - px[j2]) ** 2 + (py[j1] - py[j2]) ** 2) ** 0.5 * 0.02)

    def travelc(j1, j2):
        return (((pcx[j1] - pcx[j2]) ** 2 + (pcy[j1] - pcy[j2]) ** 2) ** 0.5 * 2.2)

    def travelp(i, j):
        if social == True:
            if ratio == True:
                return (((px[j] - xc[i]) ** 2 + (py[j] - yc[i]) ** 2) ** 0.5 * 2.8)
            else:
                return (((px[j] - xc[i]) ** 2 + (py[j] - yc[i]) ** 2) ** 0.5 * 1.5)
        else:
            return 0

    mo = Model("nvrp")
    z = {}
    for i in range(numc):  # include hospital
        z[i] = mo.addVar(vtype=GRB.BINARY, name="z[%s]" % (i))

    p1 = {}
    for i in range(numc):
        for k in range(numvr):
            p1[i, k] = mo.addVar(vtype=GRB.BINARY, name="p1[%s,%s]" % (i, k))

    p2 = {}
    for i in range(numc):
        for k in range(numnr):
            p2[i, k] = mo.addVar(vtype=GRB.BINARY, name="p2[%s,%s]" % (i, k))

    p3 = {}
    for i in range(numhr):
        p3[i] = mo.addVar(vtype=GRB.BINARY, name="p3[%s]" % (i))

    w1 = {}
    for i in range(numc):
        for j in range(nump):
            for k in range(numvr):
                w1[i, j, k] = mo.addVar(vtype=GRB.BINARY, name="w1[%s,%s,%s]" % (i, j, k))

    w2 = {}
    for i in range(numc):
        for j in range(nump):
            for k in range(numnr):
                w2[i, j, k] = mo.addVar(vtype=GRB.BINARY, name="w2[%s,%s,%s]" % (i, j, k))

    w3 = {}
    for j in range(nump):
        for k in range(numhr):
            w3[j, k] = mo.addVar(vtype=GRB.BINARY, name="w3[%s,%s]" % (j, k))

    s1 = {}
    for j in range(nump):
        for k in range(numvr):
            s1[j, k] = mo.addVar(lb=st[j], ub=ed[j], vtype=GRB.CONTINUOUS, name="s1[%s,%s]" % (j, k))

    s2 = {}
    for j in range(nump):
        for k in range(numnr):
            s2[j, k] = mo.addVar(lb=st[j], ub=ed[j], vtype=GRB.CONTINUOUS, name="s2[%s,%s]" % (j, k))

    s3 = {}
    for j in range(nump):
        for k in range(numvr):
            s3[j, k] = mo.addVar(lb=st[j], ub=ed[j], vtype=GRB.CONTINUOUS, name="s3[%s,%s]" % (j, k))

    x1 = {}
    for i in range(numc):
        for j1 in range(nump + numc):
            for j2 in range(nump + numc):
                for k in range(numvr):
                    x1[i, j1, j2, k] = mo.addVar(vtype=GRB.BINARY, name="x1[%s,%s,%s,%s]" % (i, j1, j2, k))

    x2 = {}
    for i in range(numc):
        for j1 in range(nump + numc):
            for j2 in range(nump + numc):
                for k in range(numnr):
                    x2[i, j1, j2, k] = mo.addVar(vtype=GRB.BINARY, name="x2[%s,%s,%s,%s]" % (i, j1, j2, k))

    x3 = {}
    for j1 in range(nump + numc):
        for j2 in range(nump + numc):
            for k in range(numhr):
                x3[j1, j2, k] = mo.addVar(vtype=GRB.BINARY, name="x3[%s,%s,%s]" % (j1, j2, k))

    for k in range(numvr):  # i = hospital
        mo.addConstr(p1[numc1, k] == 0)
    for k in range(numnr):
        mo.addConstr(p2[numc1, k] == 0)

    for j in range(nump1):  # j1 type nr, hr = 0
        for k in range(numnr):
            for i in range(numc):
                mo.addConstr(w2[i, j, k] == 0)
        for k in range(numhr):
            mo.addConstr(w3[j, k] == 0)

    for i in range(numc):
        for j1 in range(nump1):
            for j2 in range(nump + numc):
                for k in range(numnr):  # j1 -> all
                    mo.addConstr(x2[i, j1, j2, k] == 0)
        for j1 in range(nump1 + numc):
            for j2 in range(nump1):
                for k in range(numnr):  # all -> j1
                    mo.addConstr(x2[i, j1, j2, k] == 0)

    for j1 in range(nump1):
        for j2 in range(nump + numc):
            for k in range(numhr):  # j1 -> all
                mo.addConstr(x3[j1, j2, k] == 0)
    for j1 in range(nump1 + numc):
        for j2 in range(nump1):
            for k in range(numhr):  # all -> j1
                mo.addConstr(x3[j1, j2, k] == 0)

    for j1 in range(numc1):
        for j2 in range(nump + numc1):
            for k in range(numhr):  # c -> all
                mo.addConstr(x3[nump + j1, j2, k] == 0)

    for j2 in range(numc1):
        for j1 in range(nump + numc1):
            for k in range(numhr):  # c -> all
                mo.addConstr(x3[j1, nump + j2, k] == 0)

    cc = {}
    cc[0] = {}

    for j2 in range(nump2):
        temp = 0
        for i in range(numc1):
            for k in range(numvr):
                temp += w1[i, nump1 + j2, k]
        for i in range(numc1):
            for k in range(numnr):
                temp += w2[i, nump1 + j2, k]
        for k in range(numhr):
            temp += w3[nump1 + j2, k]
        cc[0][j2] = mo.addConstr(temp == 1, name="c0_[%s]" % (j2))

    cc[1] = {}
    for j1 in range(nump1):
        temp = 0
        for i in range(numc1):
            for k in range(numvr):
                temp += w1[i, j1, k]
        cc[1][j1] = mo.addConstr(temp == 1, name="c1_[%s]" % (j1))

    cc[2] = {}
    for k in range(numvr):
        temp = 0
        for i in range(numc1):
            temp += p1[i, k]
        cc[2][k] = mo.addConstr(temp <= 1, name="c2_[%s]" % (k))

    cc[3] = {}
    for k in range(numnr):
        temp = 0
        for i in range(numc1):
            temp += p2[i, k]
        cc[3][k] = mo.addConstr(temp <= 1, name="c3_[%s]" % (k))



    cc[4] = {}
    for k in range(numhr):
        cc[4][k] = mo.addConstr(p3[k] <= 1, name="c4_[%s]" % (k))

    cc[5] = {}
    for i in range(numc1):
        for k in range(numvr):
            cc[5][i, k] = mo.addConstr(p1[i, k] <= z[i], name="c5_[%s,%s]" % (i, k))

    cc[6] = {}
    for i in range(numc1):
        for k in range(numnr):
            cc[6][i, k] = mo.addConstr(p2[i, k] <= z[i], name="c6_[%s,%s]" % (i, k))

    cc[7] = {}
    for i in range(numc1):
        for j in range(nump):
            for k in range(numvr):
                cc[7][i, j, k] = mo.addConstr(w1[i, j, k] <= p1[i, k], name="c7_[%s,%s,%s]" % (i, j, k))

    cc[8] = {}
    for i in range(numc1):
        for j in range(nump1, nump):
            for k in range(numnr):
                cc[8][i, j, k] = mo.addConstr(w2[i, j, k] <= p2[i, k], name="c8_[%s,%s,%s]" % (i, j, k))

    cc[9] = {}
    for j in range(nump1, nump):
        for k in range(numhr):
            cc[9][j, k] = mo.addConstr(w3[j, k] <= p3[k], name="c9_[%s,%s]" % (j, k))

    # cc[10] = {}
    # for i in range(numc1):
    #    for k in range(numnr):
    #        temp = 0
    #        for j in range(nump2):
    #            temp += p2_ser[j] * w2[i, nump1 + j, k]
    #        cc[10][i, k] = mo.addConstr(temp <= 10, name="c10_[%s,%s]" %(i, k))


    # cc[11] = {}
    # for k in range(numhr):
    #    temp = 0
    #    for j in range(nump2):
    #        temp += p2_ser[j] * w3[nump1 + j, k]
    #    cc[11][k] = mo.addConstr(temp <= 10, name="c11_[%s,%s]" %(j, k))


    cc[12] = {}
    for i in range(numc1):
        for j1 in range(nump):
            for k in range(numvr):
                temp = 0
                for j2 in range(nump + numc1):
                    if j2 != j1:
                        temp += x1[i, j1, j2, k]
                cc[12][i, j1, j2, k] = mo.addConstr(temp == w1[i, j1, k], name="c12_[%s,%s,%s,%s]" % (i, j1, j2, k))

    cc[121] = {}
    for i in range(numc1):
        for j1 in range(nump):
            for k in range(numvr):
                temp = 0
                for j2 in range(nump + numc1):
                    if j2 != j1:
                        temp += x1[i, j2, j1, k]
                cc[121][i, j1, j2, k] = mo.addConstr(temp == w1[i, j1, k], name="c121_[%s,%s,%s,%s]" % (i, j1, j2, k))

    cc[13] = {}
    for i in range(numc1):
        for j1 in range(nump1, nump):
            for k in range(numnr):
                temp = 0
                for j2 in range(nump1, nump + numc1):
                    if j2 != j1:
                        temp += x2[i, j1, j2, k]
                cc[13][i, j1, j2, k] = mo.addConstr(temp == w2[i, j1, k], name="c13_[%s,%s,%s,%s]" % (i, j1, j2, k))

    cc[131] = {}
    for i in range(numc1):
        for j1 in range(nump1, nump):
            for k in range(numnr):
                temp = 0
                for j2 in range(nump1, nump + numc1):
                    if j2 != j1:
                        temp += x2[i, j2, j1, k]
                cc[131][i, j1, j2, k] = mo.addConstr(temp == w2[i, j1, k], name="c131_[%s,%s,%s,%s]" % (i, j1, j2, k))

    # set 0
    # for i in range(numc):
    #    for j1 in range(nump, nump + numc):
    #        for j2 in range(nump, nump + numc):
    #            for k in range(numnr):
    #                mo.addConstr(x2[i, j1, j2, k] == 0)


    cc[14] = {}
    for j1 in range(nump1, nump):
        for k in range(numhr):
            temp = 0
            for j2 in range(nump1, nump + 1):
                if j2 != j1:
                    temp += x3[j1, j2, k]
            cc[14][j1, j2, k] = mo.addConstr(temp == w3[j1, k], name="c14_[%s,%s,%s]" % (j1, j2, k))

    cc[141] = {}
    for j1 in range(nump1, nump):
        for k in range(numhr):
            temp = 0
            for j2 in range(nump1, nump + 1):
                if j2 != j1:
                    temp += x3[j2, j1, k]
            cc[141][j1, j2, k] = mo.addConstr(temp == w3[j1, k], name="c141_[%s,%s,%s]" % (j1, j2, k))

    cc[15] = {}
    for i in range(numc1):
        for k in range(numvr):
            for j in range(nump):
                temp1 = x1[i, nump + i, j, k]  # depot
                temp2 = x1[i, j, nump + i, k]  # depot
                for j1 in range(nump):
                    if j1 != j:
                        temp1 += x1[i, j1, j, k]
                for j2 in range(nump):
                    if j2 != j:
                        temp2 += x1[i, j, j2, k]
                cc[15][i, j, k] = mo.addConstr(temp1 == temp2, name="c15_[%s,%s,%s]" % (i, j, k))

    cc[16] = {}
    for i in range(numc1):
        for k in range(numnr):
            for j in range(nump1, nump):
                temp1 = x2[i, nump + i, j, k]
                temp2 = x2[i, j, nump + i, k]
                for j1 in range(nump1, nump):
                    if j1 != j:
                        if j1 != j:
                            temp1 += x2[i, j1, j, k]
                for j2 in range(nump1, nump):
                    if j2 != j:
                        if j2 != j:
                            temp2 += x2[i, j, j2, k]
                cc[16][i, j, k] = mo.addConstr(temp1 == temp2, name="c16_[%s,%s,%s]" % (i, j, k))

    cc[17] = {}
    for k in range(numhr):
        for j in range(nump1, nump):
            temp1 = x3[nump + numc - 1, j, k]
            temp2 = x3[j, nump + numc - 1, k]
            for j1 in range(nump1, nump):
                if j1 != j:
                    temp1 += x3[j1, j, k]
            for j2 in range(nump1, nump):
                if j2 != j:
                    temp2 += x3[j, j2, k]
            cc[17][j, k] = mo.addConstr(temp1 == temp2, name="c17_[%s,%s]" % (j, k))

        temp11 = 0
        temp22 = 0
        for j1 in range(nump1, nump):
            if j1 != j:
                temp11 += x3[j1, nump + numc1 - 1, k]
        for j2 in range(nump1, nump):
            if j2 != j:
                temp22 += x3[nump + numc1 - 1, j2, k]
        cc[17][nump1 + numc - 1, k] = mo.addConstr(temp11 == temp22, name="c17_[%s,%s]" % (nump + numc - 1, k))

    cc[18] = {}  # depot in flow = out flow
    cc[19] = {}
    for i in range(numc1):
        for k in range(numvr):
            temp1 = 0
            temp2 = 0
            for j in range(nump):
                temp1 += x1[i, nump + i, j, k]
                temp2 += x1[i, j, nump + i, k]

                # for ii in range(numc1):
                #    if ii != i:
                #        mo.addConstr(x1[i, nump + ii, j, k] == 0) # *
                #        mo.addConstr(x1[i, j, nump + ii, k] == 0) # *

            cc[18][i, k] = mo.addConstr(temp1 == p1[i, k], name="c18_[%s,%s]" % (i, k))
            cc[19][i, k] = mo.addConstr(temp2 == p1[i, k], name="c19_[%s,%s]" % (i, k))

    cc[20] = {}
    cc[21] = {}
    for i in range(numc1):
        for k in range(numnr):
            temp1 = 0
            temp2 = 0
            for j in range(nump1, nump):
                temp1 += x2[i, nump + i, j, k]
                temp2 += x2[i, j, nump + i, k]

                # for ii in range(numc1):
                #    if ii != i:
                #        mo.addConstr(x2[i, nump + ii, j, k] == 0) # *
                #        mo.addConstr(x2[i, j, nump + ii, k] == 0) # *

            cc[20][i, k] = mo.addConstr(temp1 == p2[i, k], name="c20_[%s,%s]" % (i, k))
            cc[21][i, k] = mo.addConstr(temp2 == p2[i, k], name="c21_[%s,%s]" % (i, k))

    cc[22] = {}
    cc[23] = {}

    for k in range(numhr):
        temp1 = 0
        temp2 = 0
        for j in range(nump1, nump):
            temp1 += x3[nump + numc - 1, j, k]
            temp2 += x3[j, nump + numc - 1, k]
        cc[22][k] = mo.addConstr(temp1 == p3[k], name="c22_[%s]" % (k))  # *
        cc[23][k] = mo.addConstr(temp2 == p3[k], name="c23_[%s]" % (k))  # *

    cc[24] = {}
    for i in range(numc1):
        for j1 in range(nump):
            for j2 in range(nump):
                if j1 != j2:
                    for k in range(numvr):
                        cc[24][i, j1, j2, k] = mo.addConstr(s1[j1, k] + travel(j1, j2) + ser[j1] -
                                                            100000 * (1 - x1[i, j1, j2, k]) <= s1[j2, k],
                                                            name="c24_[%s,%s,%s,%s]" % (i, j1, j2, k))

    cc[25] = {}
    for i in range(numc1):
        for j1 in range(nump1, nump):
            for j2 in range(nump1, nump):
                if j2 != j1:
                    for k in range(numnr):
                        cc[25][i, j1, j2, k] = mo.addConstr(s2[j1, k] + ser[j1] -
                                                            100000 * (1 - x2[i, j1, j2, k]) <= s2[j2, k],
                                                            name="c25_[%s,%s,%s,%s]" % (i, j1, j2, k))

    cc[26] = {}
    for j1 in range(nump1, nump):
        for j2 in range(nump1, nump):
            if j2 != j1:
                for k in range(numhr):
                    cc[26][j1, j2, k] = mo.addConstr(s3[j1, k] + ser[j1] -
                                                     100000 * (1 - x3[j1, j2, k]) <= s3[j2, k],
                                                     name="c26_[%s,%s,%s]" % (j1, j2, k))

                    # cc[27] = {}
    # for i in range(numc1):
    #    for k in range(numvr):
    #        temp = 0
    #        for j1 in range(nump1 + numc):
    #            for j2 in range(nump1 + numc):
    #                if j2 != j1:
    #                    temp += travel(j1, j2) * x1[i, j1, j2, k]
    #        for j in range(nump):
    #            temp += ser[j] * w1[i, j, k]

    # cc[27][i, k] = mo.addConstr(temp <= 10, name="c27_[%s,%s]" %(i, k))


    temp1 = 0
    temp2 = 0
    expr = LinExpr()
    for i in range(numc):
        temp1 += fee[i] * z[i]
        for k in range(numvr):
            temp1 += salary_vr[k] * p1[i, k]
        for k in range(numnr):
            temp1 += salary_nr[k] * p2[i, k]
    for k in range(numhr):
        temp1 += salary_hr[k] * p3[k]

    for i in range(numc):
        for k in range(numvr):
            for j1 in range(nump + numc):
                for j2 in range(nump + numc):
                    if j2 != j1:
                        temp1 += travelc(j1, j2) * x1[i, j1, j2, k]

    if social == True:
        for i in range(numc):
            for k in range(numnr):
                for j in range(nump):
                    temp2 += travelp(i, j) * w2[i, j, k]
                    expr.addTerms(travelp(i, j), w2[i, j, k])
        for k in range(numhr):
            for j in range(nump):
                temp2 += travelp(-1, j) * w3[j, k]
                expr.addTerms(travelp(-1, j), w3[j, k])

    obj = temp1 + temp2

    mo.update()

    mo.setObjective(obj, GRB.MINIMIZE)

    seconds1 = time.time()
    mo.optimize()
    seconds2 = time.time()

    duration = seconds2 - seconds1

    # print(duration)

    if social == True:
        if ratio == True:
            mo.write(name1 + 'social/highp/log/' + name + '.sol')
            mo.write(name1 + 'social/highp/lp/' + name + '.lp')
            mo.write(name1 + 'social/highp/mps/' + name + '.mps')
            path = name1 + 'social/highp/res/' + name + '.txt'
        else:
            mo.write(name1 + 'social/lowp/log/' + name + '.sol')
            mo.write(name1 + 'social/lowp/lp/' + name + '.lp')
            mo.write(name1 + 'social/lowp/mps/' + name + '.mps')
            path = name1 + 'social/lowp/res/' + name + '.txt'
    else:
        mo.write(name1 + 'original/log/' + name + '.sol')
        mo.write(name1 + 'original/lp/' + name + '.lp')
        mo.write(name1 + 'original/mps/' + name + '.mps')
        path = name1 + 'original/res/' + name + '.txt'

    obj = mo.getObjective().getValue()
    
    obj2 = 0
    if social == True:
        for i in range(numc):
            for k in range(numnr):
                for j in range(nump):
                    obj2 += travelp(i, j) * w2[i, j, k].X
        for k in range(numhr):
            for j in range(nump):
                obj2 += travelp(-1, j) * w3[j, k].X


    f = open(path, "w+")

    f.write('Time: {} \n'.format(duration))
    f.write('Objective value: {} \n'.format(obj))
    if social == True:
        f.write('Objective value for first term: {} \n'.format(obj - obj2))

    f.write("\n")
    f.write("============================================================================== \n")
    temp = 'Open clinic: '
    for i in range(numc):
        if z[i].X > 0:
            temp += str(i) + ' '
    temp += '\n'
    f.write(temp)

    f.write("\n")
    f.write("============================================================================== \n")
    f.write('Nurse Assignment: \n')
    f.write('  -> Visiting Nurse Assignment: \n')

    for i in range(numc):
        for k in range(numvr):
            if p1[i, k].X > 0:
                f.write('    Visiting nurse {} is assigned to clinic {}. \n'.format(k, i))
                for j in range(nump):
                    if w1[i, j, k].X > 0:
                        f.write('      Patient {} is assigned to visiting nurse {} in clinic {}. \n'.format(j, k, i))
                        flag = False
                        for j2 in range(nump):
                            if x1[i, j, j2, k].X > 0:
                                f.write(
                                    '        Starting time for patient {} is {}; From patient {} to patient {}.\n'.format(j,
                                                                                                                          s1[
                                                                                                                              j, k].X,
                                                                                                                          j,
                                                                                                                          j2))
                                flag = True
                        if flag == False:
                            f.write('        Starting time for patient {} is {}; From patient {} to clinic {}.\n'.format(j,
                                                                                                                         s1[
                                                                                                                             j, k].X,
                                                                                                                         j,
                                                                                                                         i))

    f.write('  -> Clinic Nurse Assignment: \n')
    for i in range(numc):
        for k in range(numnr):
            if p2[i, k].X > 0:
                f.write('    Clinic nurse {} is assigned to clinic {}. \n'.format(k, i))
                for j in range(nump):
                    if w2[i, j, k].X > 0:
                        f.write('      Patient {} is assigned to clinic nurse {} in clinic {}. \n'.format(j, k, i))
                        flag = False
                        for j2 in range(nump):
                            if x2[i, j, j2, k].X > 0:
                                f.write(
                                    '        Starting time for patient {} is {}; From patient {} to patient {}.\n'.format(j,
                                                                                                                          s2[
                                                                                                                              j, k].X,
                                                                                                                          j,
                                                                                                                          j2))
                                flag = True
                        if flag == False:
                            f.write('        Starting time for patient {} is {}; From patient {} to clinic {}.\n'.format(j,
                                                                                                                         s2[
                                                                                                                             j, k].X,
                                                                                                                         j,
                                                                                                                         i))

    f.write('  -> Hospital Nurse Assignment: \n')
    for k in range(numhr):
        # print(p3[k].X)
        if p3[k].X > 0:
            f.write('    Hospital nurse {} is assigned to hospital. \n'.format(k))
            for j in range(nump):
                if w3[j, k].X > 0:
                    f.write('      Patient {} is assigned to hospital nurse {} in hospital. \n'.format(j, k))
                    flag = False
                    for j2 in range(nump):
                        if x3[j, j2, k].X > 0:
                            f.write('        Starting time for patient {} is {}; From patient {} to patient {}.\n'.format(j,
                                                                                                                          s3[
                                                                                                                              j, k].X,
                                                                                                                          j,
                                                                                                                          j2))
                            flag = True
                    if flag == False:
                        f.write('        Starting time for patient {} is {}; From patient {} to hospital.\n'.format(j, s3[
                            j, k].X, j))
    f.flush()
    f.close()

    objj = obj - obj2
    
    if (mo.status != 2):
        return 'inf', duration, objj
    else:
        return str(round(obj, 2)), str(round(duration, 2)), str(round(objj, 2))

def plot(x, y, txt, x2, y2, txt2, x1, y1, txt3, name1, name):
    fig = plt.figure()
    plt.figure(figsize=(12, 12))
    l1 = plt.scatter(x,y, s = 20, color = 'red')
    for i, t in enumerate(txt):
        plt.annotate(t, (x[i], y[i]), xytext =(x[i]+0.5, y[i]+0.5))

    l11 = plt.scatter(x2,y2, s = 20, marker = 'x', color = 'red')
    for i, t in enumerate(txt2):
        plt.annotate(t, (x2[i], y2[i]), xytext =(x2[i]+0.5, y2[i]+0.5))

    xx1 = []
    yy1 = []
    for i in range(len(x1)):
        if i < len(x1) - 1:
            xx1.append(x1[i])
            yy1.append(y1[i])

    l2 = plt.scatter(xx1,yy1, s = 60, color="green")

    l3 = plt.scatter(x1[-1], y1[-1], s=60, color='blue')
    for i, t in enumerate(txt3):
        plt.annotate(t, (x1[i], y1[i]), xytext =(x1[i]+0.5, y1[i]+0.5))
    plt.title("Distribution of patients, clinics, and hospital")
    plt.legend((l2, l1,l11, l3), ("clinic", "patient(at home)","patient(no preference)", "hospital"))

    plt.savefig(name1 + 'fig/' + name + '.png')




filename1 = ['/blue/xiang.zhong/yutan/code/nvrp/original/', '/blue/xiang.zhong/yutan/code/nvrp/highSetup/', '/blue/xiang.zhong/yutan/code/nvrp/largeTW/']

filename2 = 'data/'
L = [10, 20, 30, 40]

with open('res.csv', 'w', newline ='') as csvfile:
    filen = ['instance', 'original no social result', 'original no social time', 'original social low p result', 'original social low p result2', 'original social low p time', 'original social high p result', 'original social high p result2', 'original social high p time', 'highSetup no social result', 'highSetup no social time', 'highSetup social low p result', 'highSetup social low p result2', 'highSetup social low p time', 'highSetup social high p result', 'highSetup social high p result2', 'highSetup social high p time', 'largeTW no social result', 'largeTW no social time', 'largeTW social low p result', 'largeTW social low p result2', 'largeTW social low p time', 'largeTW social high p result', 'largeTW social high p result2', 'largeTW social high p time']
    writer = csv.DictWriter(csvfile, fieldnames = filen)
    writer.writeheader()
    
    for first in L:
        for second in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            file = str(first) + '/' + str(second)
            
            opt_nosocial, duration_nosocial, opt_nosocial1 = gurobiSolve(filename1[0], 'data/', file, False, False) # no social
            opt_lowp, duration_lowp, opt_lowp1 = gurobiSolve(filename1[0], 'data/', file, True, False) # social, low p
            opt_highp, duration_highp, opt_highp1 = gurobiSolve(filename1[0], 'data/', file, True, True) # social, high p
            
            
            opt2_nosocial, duration2_nosocial, opt2_nosocial1 = gurobiSolve(filename1[1], 'data/', file, False, False) # no social
            opt2_lowp, duration2_lowp, opt2_lowp1 = gurobiSolve(filename1[1], 'data/', file, True, False) # social, low p
            opt2_highp, duration2_highp, opt2_highp1 = gurobiSolve(filename1[1], 'data/', file, True, True) # social, high p
            
            
            opt3_nosocial, duration3_nosocial, opt3_nosocial1 = gurobiSolve(filename1[2], 'data/', file, False, False) # no social
            opt3_lowp, duration3_lowp, opt3_lowp1 = gurobiSolve(filename1[2], 'data/', file, True, False) # social, low p
            opt3_highp, duration3_highp, opt3_highp1 = gurobiSolve(filename1[2], 'data/', file, True, True) # social, high p
                
                
            writer.writerow({'instance': file, 'original no social result': opt_nosocial, 'original no social time': duration_nosocial, 'original social low p result': opt_lowp, 'original social low p result2': opt_lowp1, 'original social low p time': duration_lowp, 'original social high p result': opt_highp, 'original social high p result2': opt_highp1, 'original social high p time': duration_highp, 'highSetup no social result': opt2_nosocial, 'highSetup no social time': duration2_nosocial, 'highSetup social low p result': opt2_lowp, 'highSetup social low p result2': opt2_lowp1, 'highSetup social low p time': duration2_lowp, 'highSetup social high p result': opt2_highp, 'highSetup social high p result2': opt2_highp1, 'highSetup social high p time': duration2_highp, 'largeTW no social result': opt3_nosocial, 'largeTW no social time': duration3_nosocial, 'largeTW social low p result': opt3_lowp, 'largeTW social low p result2': opt3_lowp1, 'largeTW social low p time': duration3_lowp, 'largeTW social high p result': opt3_highp, 'largeTW social high p result2': opt3_highp1, 'largeTW social high p time': duration3_highp})
            csvfile.flush()
            
    
    

            
            



