import torch
from ALG_HEFT import HEFT


class taskGenerator(object):
    def __init__(self, N, P):
        self.N = N
        self.P = P

        edgeProbality = 0.5
        ccr = [5, 30]
        wr = [5, 30]
        pricer = [3, 10]
        ddlr = [1.1, 2]

        self.w = torch.rand(N, P) * (wr[1] - wr[0]) + wr[0]
        self.price = torch.rand(P, 1) * (pricer[1] - pricer[0]) + pricer[0]

        cost_min = (torch.min(self.w * (self.price.reshape((1, self.P))), 1).values).reshape(self.N, 1)
        cost_max = (torch.max(self.w * (self.price.reshape((1, self.P))), 1).values).reshape(self.N, 1)
        cost_min_G = torch.sum(cost_min).item()
        cost_max_G = torch.sum(cost_max).item()
        self.cost_bud = torch.rand(1).item() * (cost_max_G - cost_min_G) + cost_min_G

        self.c = torch.zeros(N, N) - 1
        mode = True
        for i in range(N - 2, 0, -1):
            if mode:
                for j in range(i + 1, N):
                    if (torch.rand(1).item() < edgeProbality):
                    	self.c[i, j] = round(torch.rand(1).item() * (ccr[1] - ccr[0]) + ccr[0])
            else:
                for j in range(i + 1, N):
                    if (torch.rand(1).item() < edgeProbality):
                        self.c[j, i] = round(torch.rand(1).item() * (ccr[1] - ccr[0]) + ccr[0])
            mode = ~mode

        for i in range(1, N - 1):
            if (torch.max(self.c[:, i]) == -1):
            	self.c[0, i] = round(torch.rand(1).item() * (ccr[1] - ccr[0]) + ccr[0])
            if (torch.max(self.c[i, :]) == -1):
                self.c[i, N - 1] = round(torch.rand(1).item() * (ccr[1] - ccr[0]) + ccr[0])

        self.HEFTSchd = HEFT(self.N, self.P, self.c, self.w)
        self.ddl = self.HEFTSchd.taskRecord[:, 1].max().item() * (torch.rand(1).item() * (ddlr[1] - ddlr[0]) + ddlr[0])