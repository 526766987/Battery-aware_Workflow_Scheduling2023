import torch
from budgetValue import budgetValue
from taskGenerator import taskGenerator


class sampleGenerator(object):
    def __init__(self, N, P, N_max, P_max):
        self.N_max = N_max
        self.P_max = P_max

        workflow = taskGenerator(N, P)

        example = budgetValue(workflow.N, workflow.P, workflow.c, workflow.w, workflow.price, workflow.ddl)
        example.budgetCal()

        self.makeDataSet(workflow, example)

    def makeDataSet(self, workflow, example):
        if (workflow.N < self.N_max):
            workflow.c = torch.cat([workflow.c, torch.zeros(workflow.N, self.N_max - workflow.N) - 1], dim = 1)
            workflow.c = torch.cat([workflow.c, torch.zeros(self.N_max - workflow.N, self.N_max) - 1], dim = 0)
        if (workflow.P < self.P_max):
            workflow.w = torch.cat([workflow.w, torch.zeros(workflow.N, self.P_max - workflow.P) - 1], dim = 1)
            workflow.price = torch.cat([workflow.price, torch.zeros(self.P_max - workflow.P, 1) - 1], dim = 0)
        if (workflow.N < self.N_max):
         	workflow.w = torch.cat([workflow.w, torch.zeros(self.N_max - workflow.N, self.P_max) - 1], dim = 0)

        self.Input1 = workflow.w.reshape(1, -1)
        self.Input2 = workflow.c.reshape(1, -1)
        self.Input3 = workflow.price.reshape(1, -1)
        self.Input4 = torch.tensor([[workflow.N, workflow.P, workflow.ddl]])
        self.Output = torch.tensor([[(example.cost_bud - example.cost_min_G) / (example.cost_max_G - example.cost_min_G)]])
        self.Prep = torch.tensor([[example.cost_min_G, example.cost_max_G]])