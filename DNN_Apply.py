import torch
from DNN_Train import Net

class DNN_Net_Solution(object):
    def __init__(self):
        self.N_max = 100
        self.P_max = 10

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net = Net(self.N_max * self.P_max, self.N_max * self.N_max, self.P_max, 3, 1)
        self.net.to(self.device)

        self.net.load_state_dict(torch.load("./myModel.pth", map_location=torch.device(self.device)))
        self.net.eval()

    def apply(self, N, P, c, w, price, ddl, cost_min_G, cost_max_G):
        if (N < self.N_max):
            c = torch.cat([c, torch.zeros(N, self.N_max - N) - 1], dim = 1)
            c = torch.cat([c, torch.zeros(self.N_max - N, self.N_max) - 1], dim = 0)
        if (P < self.P_max):
            w = torch.cat([w, torch.zeros(N, self.P_max - P) - 1], dim = 1)
            price = torch.cat([price, torch.zeros(self.P_max - P, 1) - 1], dim = 0)
        if (N < self.N_max):
             w = torch.cat([w, torch.zeros(self.N_max - N, self.P_max) - 1], dim = 0)

        Input1 = w.reshape(1, -1)
        Input2 = c.reshape(1, -1)
        Input3 = price.reshape(1, -1)
        Input4 = torch.tensor([[N, P, ddl]])
        Prep = torch.tensor([[cost_min_G, cost_max_G]])

        Input1 = Input1.to(self.device)
        Input2 = Input2.to(self.device)
        Input3 = Input3.to(self.device)
        Input4 = Input4.to(self.device)
        Prep = Prep.to(self.device)

        solu = self.net(Input1, Input2, Input3, Input4)
        bud = solu * (Prep[:, 1].reshape(-1, 1) - Prep[:, 0].reshape(-1, 1)) + Prep[:, 0].reshape(-1, 1)
        return bud
