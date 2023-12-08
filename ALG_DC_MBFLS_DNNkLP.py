import docplex.mp.model as cpx
import cplex
import torch
import math
import time
import gc
import os


class DC_MBFLS_DNNkLP(object):
    def __init__(self, N, P, c, w, price, ddl, net):

        time_start = time.clock()

        self.N = N
        self.P = P
        self.c = c  # N*N
        self.w = w  # N*P
        self.price = price  # P*1
        self.ddl = ddl
        self.net = net # DNN

        # Init
        self.cost_min = torch.zeros(N, 1)  # N*1
        self.cost_max = torch.zeros(N, 1)  # N*1
        self.cost_min_G = 0
        self.cost_max_G = 0
        self.ranku = torch.zeros(N, 1) - 1  # N*1
        self.cost_bud = 0
        self.ClpexNone = False

        self.scheduler()

        time_end = time.clock()
        self.useTime = time_end - time_start

    def scheduler(self):
        self.budgetCal()
        self.stage2()

    def budgetCal(self):
        # cost_min & cost_max
        self.cost_min = (torch.min(self.w * (self.price.reshape((1, self.P))), 1).values).reshape(self.N, 1)
        self.cost_max = (torch.max(self.w * (self.price.reshape((1, self.P))), 1).values).reshape(self.N, 1)
        self.cost_min_G = torch.sum(self.cost_min).item()
        self.cost_max_G = torch.sum(self.cost_max).item()

        # rank
        self.rank_u_ni(0)
        self.ranku, order = torch.sort(self.ranku, dim=0, descending=True)
        self.ranku = torch.cat((self.ranku, order), dim=1)

        # try DNN
        self.cost_bud = self.net.apply(self.N, self.P, self.c, self.w, self.price, self.ddl, self.cost_min_G, self.cost_max_G)
        self.stage1()
        if (self.ddl >= torch.max(self.stage1_taskRecord[:, 1])):
            return

        # IF DNN failed
        # Maximum Budget Verification
        self.cost_bud = self.cost_max_G
        self.stage1()
        if (self.ddl < torch.max(self.stage1_taskRecord[:, 1])):
            self.taskRecord = self.stage1_taskRecord
            self.processRecord = self.stage1_processRecord
            return

        # set endpoints
        c1 = self.cost_min_G + 1
        c2 = self.cost_max_G
        temp = 3

        # start
        while (abs(c2 - c1) > 1):
            # 试一下cost_bud
            self.cost_bud = round((c1 + c2) / 2)
            self.stage1()
            temp = temp - 1
            if (torch.max(self.stage1_taskRecord[:, 1]) <= self.ddl):
                if (temp == 0):
                    break
                else:
                    c2 = self.cost_bud
            else:
                if (temp == 0):
                    self.cost_bud = round(c2)
                    self.stage1()
                    break
                else:
                    c1 = self.cost_bud
        return

    def rank_u_ni(self, i):
        maxSucc = 0
        for j in range(0, self.N):
            if (self.c[i, j] >= 0):
                if (self.ranku[j, 0] == -1):
                    self.rank_u_ni(j)
                if (self.c[i, j] + self.ranku[j, 0] > maxSucc):
                    maxSucc = self.c[i, j] + self.ranku[j, 0]
        self.ranku[i, 0] = torch.mean(self.w[i, :]) + maxSucc

    def stage1(self):
        # check
        if (self.cost_bud <= self.cost_min_G):
            return

        # bl
        bl = (self.cost_bud - self.cost_min_G) / (self.cost_max_G - self.cost_min_G)
        cost_bl = self.cost_min + (self.cost_max - self.cost_min) * bl

        # reserved budget
        RB = self.cost_bud
        RCB = self.cost_bud

        # Storage variables for results
        avail = torch.zeros(self.P)
        taskRecord = torch.zeros(self.N, 3)
        processRecord = [torch.tensor([[]]) for i in range(self.P)]
        processRecordCount = torch.zeros(self.P)

        # Find EST
        for i in self.ranku[:, 1]:
            i = round(i.item())
            RCB = RCB - cost_bl[i, 0].item()
            EST = torch.zeros(self.P)
            EFT = torch.zeros(self.P)
            for j in range(0, self.P):
                # Cal readytime
                readytime = 0
                for m in self.ranku[:, 1]:
                    m = round(m.item())
                    if (m == i):
                        break
                    if (self.c[m, i] >= 0):
                        if (taskRecord[m, 2] == j):
                            if (readytime <= taskRecord[m, 1]):
                                readytime = taskRecord[m, 1]
                        else:
                            if (readytime <= taskRecord[m, 1] + self.c[m, i]):
                                readytime = taskRecord[m, 1] + self.c[m, i]
                # try insert
                haveInsert = False
                if (processRecordCount[j].item() == 1):
                    if (readytime + self.w[i, j] <= processRecord[j][0, 0].item()):
                        haveInsert = True
                        EST[j] = readytime
                elif (processRecordCount[j].item() > 1):
                    if (readytime + self.w[i, j] <= processRecord[j][0, 0].item()):
                        haveInsert = True
                        EST[j] = readytime
                    else:
                        for k in range(0, round(processRecordCount[j].item() - 1)):
                            if ((processRecord[j][k, 1].item() >= readytime) & (processRecord[j][k, 1].item() + self.w[i,j] <= processRecord[j][k+1, 0].item())):
                                haveInsert = True
                                EST[j] = processRecord[j][k, 1]
                                break
                            elif ((processRecord[j][k, 1].item() <= readytime) & (readytime + self.w[i,j] <= (processRecord[j][k+1, 0].item()))):
                                haveInsert = True
                                EST[j] = readytime
                                break
                if haveInsert is False:
                    EST[j] = max(avail[j].item(), readytime)
                EFT[j] = EST[j] + self.w[i, j]

            p_sel = -1
            for j in range(0, self.P):
                if (self.w[i, j] * self.price[j] <= (RB - RCB)):
                    if (p_sel == -1):
                        p_sel = j
                    elif (EFT[p_sel] > EFT[j]):
                        p_sel = j

            if (p_sel == -1):
                for j in range(0, self.P):
                    if (p_sel == -1):
                        p_sel = j
                    elif (EFT[p_sel] > EFT[j]):
                        p_sel = j

            # write in processRecord
            if (processRecordCount[p_sel].item() == 0):
                processRecord[p_sel] = torch.tensor([[EST[p_sel].item(), EFT[p_sel].item(), i]])
            else:
                processRecord[p_sel] = torch.cat((processRecord[p_sel], torch.tensor([[EST[p_sel].item(), EFT[p_sel].item(), i]])), dim = 0)
            processRecordCount[p_sel] = processRecordCount[p_sel] + 1
            if (processRecordCount[p_sel].item() > 1):
                _, original_indices = processRecord[p_sel][:, 0].sort()
                processRecord[p_sel] = processRecord[p_sel][original_indices]
            # write in taskRecord
            taskRecord[i, 0] = EST[p_sel]
            taskRecord[i, 1] = EFT[p_sel]
            taskRecord[i, 2] = p_sel
            # refresh availtime
            avail[p_sel] = EFT[p_sel]
            RB = RB - self.w[i, p_sel] * self.price[p_sel]

        # Output
        self.stage1_taskRecord = taskRecord
        self.stage1_processRecord = processRecord

    def stage2(self):
        self.stage2_taskRecord = self.stage1_taskRecord
        self.stage2_processRecord = self.stage1_processRecord

        # integer programming auxiliary variable
        y = torch.zeros(self.N, self.P, math.ceil(self.ddl) + 1, dtype=torch.bool)
        for i in range(0, self.N):
            k = round(self.stage2_taskRecord[i, 2].item())
            for t in range(round(self.stage2_taskRecord[i, 0].item()), round(self.stage2_taskRecord[i, 1].item())):
                y[i, k, t] = True

        # reverse order
        for i in torch.flip(self.ranku[:, 1], dims = [0]):
            i = round(i.item())
            p_sel = round(self.stage2_taskRecord[i, 2].item())
            temp_i = (self.stage2_processRecord[p_sel][:, 2] == i).nonzero().item()
            rce = self.ddl
            for m in torch.flip(self.ranku[:, 1], dims = [0]):
                m = round(m.item())
                if (m == i):
                    break
                if (self.c[i, m] >= 0):
                    if (self.stage2_taskRecord[i, 2] == self.stage2_taskRecord[m, 2]):
                        if (rce >= (self.stage2_taskRecord[m, 0] - self.w[i, p_sel])):
                            rce = self.stage2_taskRecord[m, 0] - self.w[i, p_sel]
                    else:
                        if (rce >= (self.stage2_taskRecord[m, 0] - self.c[i, m] - self.w[i, p_sel])):
                            rce = self.stage2_taskRecord[m, 0] - self.c[i, m] - self.w[i, p_sel]
            if ((rce != self.ddl) & (rce > self.stage2_taskRecord[i, 0])):
                startTime = self.stage2_taskRecord[i, 0].item()
                if (math.floor(rce) - math.ceil(startTime) <= 0):
                    continue
                    
                y[i, p_sel, :] = False

                # integer programming
                # scheduling space [startTime, rce]
                # sta space [startTime, endTime)
                startTime = math.ceil(startTime)
                rce = math.floor(rce)
                endTime = rce + self.w[i, p_sel].item()
                endTime = math.ceil(endTime)
                
                # IP Modeling
                opt_model = cpx.Model()
                x = {t: opt_model.binary_var(name="{0}".format(t)) for t in range(0, rce + 1 - startTime)}

                opt_model.add_constraint(\
                    ct = (opt_model.sum(x[t] for t in range(0, rce + 1 - startTime)) == 1))
                
                for tc in range(startTime, endTime):
                    opt_model.add_constraint(\
                        ct = (((opt_model.sum(x[t]\
                            for t in range(max(0, math.ceil(tc - self.w[i, p_sel].item() + 1 - startTime)),\
                            min(tc + 1 - startTime, rce + 1 - startTime))) + torch.sum(y[:, p_sel, tc]).item()) <= 1)
                        ))

                objective = opt_model.minimize(\
                    opt_model.sum_squares(\
                        opt_model.sum(x[t] * self.price[p_sel, 0].item()\
                            for t in range(max(0, math.ceil(tc - self.w[i, p_sel].item() + 1 - startTime)),\
                            min(tc + 1 - startTime, rce + 1 - startTime))) +\
                        torch.sum(y[:, :, tc] * self.price[:, 0]).item()\
                        for tc in range(startTime, endTime)\
                    ))

                # Solve
                solution = opt_model.solve(agent='local')
                # Get solution
                if (solution is None):
                    self.ClpexNone = True
                    del opt_model
                    gc.collect()
                    continue
                for val in solution.iter_var_values():
                    self.stage2_taskRecord[i, 0] = int(str(val[0])) + math.ceil(startTime)
                    self.stage2_taskRecord[i, 1] = self.stage2_taskRecord[i, 0] + self.w[i, p_sel].item()
                    self.stage2_processRecord[p_sel][temp_i, 0] = self.stage2_taskRecord[i, 0]
                    self.stage2_processRecord[p_sel][temp_i, 1] = self.stage2_taskRecord[i, 1]
                    for t in range(round(self.stage2_taskRecord[i, 0].item()), round(self.stage2_taskRecord[i, 1].item())):
                        y[i, p_sel, t] = True

                del opt_model
                gc.collect()

            elif (rce == self.ddl):  # the exit task
                self.stage2_taskRecord[i, 0] = self.ddl - self.w[i, p_sel]
                self.stage2_taskRecord[i, 1] = self.ddl
                self.stage2_processRecord[p_sel][temp_i, 0] = self.stage2_taskRecord[i, 0]
                self.stage2_processRecord[p_sel][temp_i, 1] = self.stage2_taskRecord[i, 1]
            #else: # no space

        # Output
        #self.stage2_taskRecord
        #self.stage2_processRecord

    def SlidingConflictCheck(self, i, t):
        for k in range(0, self.N):
            if (k == i):
                continue
            if (self.stage2_taskRecord[i, 2] != self.stage2_taskRecord[k, 2]):
                continue
            if ((self.stage2_taskRecord[i, 1] + t <= self.stage2_taskRecord[k, 0]) | (self.stage2_taskRecord[k, 1] <= self.stage2_taskRecord[i, 0] + t)):
                continue
            return False
        return True

    def VolatilityCal(self, i, t, rce):
        startTime = self.stage2_taskRecord[i, 0].item()
        endTime = rce + self.w[i, round(self.stage2_taskRecord[i, 2].item())].item()
        T = range(math.floor(startTime), math.ceil(endTime + 1))
        squSum = 0
        for r in range(math.floor(startTime), math.ceil(endTime + 1)):
            localSum = 0
            for k in range(0, self.N):
                if ((k == i) & (self.stage2_taskRecord[k, 0] + t <= r) & (r < self.stage2_taskRecord[k, 1] + t)):
                    localSum = localSum + self.price[round(self.stage2_taskRecord[k, 2].item())]
                elif ((self.stage2_taskRecord[k, 0] <= r) & (r < self.stage2_taskRecord[k, 1])):
                    localSum = localSum + self.price[round(self.stage2_taskRecord[k, 2].item())]
            squSum = squSum + localSum * localSum
        return squSum