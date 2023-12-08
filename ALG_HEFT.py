import torch
import time


class HEFT(object):
    def __init__(self, N, P, c, w):

        time_start = time.clock()

        self.N = N
        self.P = P
        self.c = c
        self.w = w

        self.scheduler()
        
        time_end = time.clock()
        self.useTime = time_end - time_start

    def scheduler(self):
        self.ranku = torch.zeros(self.N, 1) - 1
        self.rank_u_ni(0)

        self.ranku, order = torch.sort(self.ranku, dim=0, descending=True)
        self.ranku = torch.cat((self.ranku, order), dim=1)

        avail = torch.zeros(self.P)
        taskRecord = torch.zeros(self.N, 3)
        processRecord = [torch.tensor([[]]) for i in range(self.P)]
        processRecordCount = torch.zeros(self.P)
        for i in self.ranku[:, 1]:
            i = round(i.item())
            EST = torch.zeros(self.P)
            EFT = torch.zeros(self.P)
            for j in range(0, self.P):
                readytime = 0
                for m in self.ranku[:, 1]:
                    m = round(m.item())
                    if (m == i):
                        break
                    if (self.c[m, i] >= 0):
                        if (taskRecord[m, 2] == j):
                            if (readytime < taskRecord[m, 1]):
                                readytime = taskRecord[m, 1]
                        else:
                            if (readytime < taskRecord[m, 1] + self.c[m, i]):
                                readytime = taskRecord[m, 1] + self.c[m, i]
                haveInsert = False
                if (processRecordCount[j].item() == 1):
                    if (self.w[i, j] + readytime < processRecord[j][0, 0].item()):
                        haveInsert = True
                        EST[j] = readytime
                elif (processRecordCount[j].item() > 1):
                    if (self.w[i, j] + readytime < processRecord[j][0, 0].item()):
                        haveInsert = True
                        EST[j] = readytime
                    else:
                        for k in range(0, round(processRecordCount[j].item() - 1)):
                            if ((processRecord[j][k, 1].item() >= readytime) & (processRecord[j][k, 1].item() + self.w[i,j] < processRecord[j][k+1, 0].item())):
                                haveInsert = True
                                EST[j] = processRecord[j][k, 1]
                                break
                            elif ((processRecord[j][k, 1].item() <= readytime) & (readytime + self.w[i,j] < (processRecord[j][k+1, 0].item()))):
                                haveInsert = True
                                EST[j] = readytime
                                break
                if haveInsert is False:
                    EST[j] = max(avail[j].item(), readytime)
                EFT[j] = EST[j] + self.w[i, j]

            p_sel = -1
            for j in range(0, self.P):
                if (p_sel == -1):
                    p_sel = j
                elif (EFT[p_sel] > EFT[j]):
                    p_sel = j

            if (processRecordCount[p_sel].item() == 0):
                processRecord[p_sel] = torch.tensor([[EST[p_sel].item(), EFT[p_sel].item(), i]])
            else:
                processRecord[p_sel] = torch.cat((processRecord[p_sel], torch.tensor([[EST[p_sel].item(), EFT[p_sel].item(), i]])), dim = 0)
            processRecordCount[p_sel] = processRecordCount[p_sel] + 1
            if (processRecordCount[p_sel].item() > 1):
                _, original_indices = processRecord[p_sel][:, 0].sort()
                processRecord[p_sel] = processRecord[p_sel][original_indices]
            taskRecord[i, 0] = EST[p_sel]
            taskRecord[i, 1] = EFT[p_sel]
            taskRecord[i, 2] = p_sel
            avail[p_sel] = EFT[p_sel]

        self.taskRecord = taskRecord
        self.processRecord = processRecord

    def rank_u_ni(self, i):
        maxSucc = 0
        for j in range(0, self.N):
            if (self.c[i, j] >= 0):
                if (self.ranku[j, 0] == -1):
                    self.rank_u_ni(j)
                if (self.c[i, j] + self.ranku[j, 0] > maxSucc):
                    maxSucc = self.c[i, j] + self.ranku[j, 0]
        self.ranku[i, 0] = torch.mean(self.w[i, :]) + maxSucc