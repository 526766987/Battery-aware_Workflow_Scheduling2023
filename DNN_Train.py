import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.distributed as dist
import torch.utils.data as data_utils
from sampleGenerator import sampleGenerator

class Net(nn.Module):
    def __init__(self, n_input1, n_input2, n_input3, n_input4, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input1, 1024)  # w
        self.hidden2 = nn.Linear(n_input3, 1024)  # price
        self.hidden3 = nn.Linear(n_input2, 1024)  # c

        self.hidden4 = nn.Linear(1024 + 1024, 512)  # w & price
        self.hidden5 = nn.Linear(1024 + 512, 512)  # w & price & c
        self.hidden6 = nn.Linear(512 + n_input4, 64)  # w & price & c & KEYDATA

        self.predict = nn.Linear(64, n_output)

    def forward(self, input1, input2, input3, input4):
        out11 = self.hidden1(input1)  # w
        out12 = torch.sigmoid(out11)

        out21 = self.hidden2(input3)  # price
        out22 = torch.sigmoid(out21)

        out31 = self.hidden3(input2)  # c
        out32 = torch.sigmoid(out31)

        out13 = torch.cat((out12, out22), dim=1)  # w & price
        out14 = self.hidden4(out13)
        out15 = torch.sigmoid(out14)

        out16 = torch.cat((out15, out32), dim=1)  # w & price & c
        out17 = self.hidden5(out16)
        out18 = torch.sigmoid(out17)

        out19 = torch.cat((out18, input4), dim=1)  # w & price & c & KEYDATA
        out1A = self.hidden6(out19)
        out1B = torch.sigmoid(out1A)

        out1C = self.predict(out1B)
        out1D = torch.sigmoid(out1C)

        return out1D

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    N_max = 100
    P_max = 10

    net = Net(N_max * P_max, N_max * N_max, P_max, 3, 1)
    net.to(device)

    # DataSelectMode(0: Readin, 1: Generate)
    selectMode = False

    if selectMode:
        trainAmount = 4000
        testAmount = 100

        TrainX_1 = torch.tensor([[]])
        TrainX_2 = torch.tensor([[]])
        TrainX_3 = torch.tensor([[]])
        TrainX_4 = torch.tensor([[]])
        TrainY = torch.tensor([[]])
        TrainPrep = torch.tensor([[]])
        for i in range(0, trainAmount):
            N = round(torch.rand(1).item() * (30 - 10) + 10)
            P = 3
            example = sampleGenerator(N, P, N_max, P_max)
            if (i == 0):
                TrainX_1 = example.Input1
                TrainX_2 = example.Input2
                TrainX_3 = example.Input3
                TrainX_4 = example.Input4
                TrainY = example.Output
                TrainPrep = example.Prep
            else:
                TrainX_1 = torch.cat((TrainX_1, example.Input1), dim = 0)
                TrainX_2 = torch.cat((TrainX_2, example.Input2), dim = 0)
                TrainX_3 = torch.cat((TrainX_3, example.Input3), dim = 0)
                TrainX_4 = torch.cat((TrainX_4, example.Input4), dim = 0)
                TrainY = torch.cat((TrainY, example.Output), dim = 0)
                TrainPrep = torch.cat((TrainPrep, example.Prep), dim = 0)

        TestX_1 = torch.tensor([[]])
        TestX_2 = torch.tensor([[]])
        TestX_3 = torch.tensor([[]])
        TestX_4 = torch.tensor([[]])
        TestY = torch.tensor([[]])
        TestPrep = torch.tensor([[]])

        for i in range(0, testAmount):
            N = round(torch.rand(1).item() * (30 - 10) + 10)
            P = 3
            example = sampleGenerator(N, P, N_max, P_max)
            if (i == 0):
                TestX_1 = example.Input1
                TestX_2 = example.Input2
                TestX_3 = example.Input3
                TestX_4 = example.Input4
                TestY = example.Output
                TestPrep = example.Prep
            else:
                TestX_1 = torch.cat((TestX_1, example.Input1), dim = 0)
                TestX_2 = torch.cat((TestX_2, example.Input2), dim = 0)
                TestX_3 = torch.cat((TestX_3, example.Input3), dim = 0)
                TestX_4 = torch.cat((TestX_4, example.Input4), dim = 0)
                TestY = torch.cat((TestY, example.Output), dim = 0)
                TestPrep = torch.cat((TestPrep, example.Prep), dim = 0)

        saveData = {'trainAmount': trainAmount, 'testAmount': testAmount, \
        'TrainX_1': TrainX_1, 'TrainX_2': TrainX_2, 'TrainX_3': TrainX_3, 'TrainX_4': TrainX_4, \
        'TrainY': TrainY, 'TrainPrep': TrainPrep, \
        'TestX_1': TestX_1, 'TestX_2': TestX_2, 'TestX_3': TestX_3, 'TestX_4': TestX_4, \
        'TestY': TestY, 'TestPrep': TestPrep}

        torch.save(saveData, "./myData.pth")

        TrainX_1 = TrainX_1.to(device)
        TrainX_2 = TrainX_2.to(device)
        TrainX_3 = TrainX_3.to(device)
        TrainX_4 = TrainX_4.to(device)
        TrainY = TrainY.to(device)
        TrainPrep = TrainPrep.to(device)

        TestX_1 = TestX_1.to(device)
        TestX_2 = TestX_2.to(device)
        TestX_3 = TestX_3.to(device)
        TestX_4 = TestX_4.to(device)
        TestY = TestY.to(device)
        TestPrep = TestPrep.to(device)

        print('DataSet Ready!')

    else:
        saveData = torch.load("./myData.pth")

        trainAmount = saveData['trainAmount']
        testAmount = saveData['testAmount']
        TrainX_1 = saveData['TrainX_1'].to(device)
        TrainX_2 = saveData['TrainX_2'].to(device)
        TrainX_3 = saveData['TrainX_3'].to(device)
        TrainX_4 = saveData['TrainX_4'].to(device)
        TrainY = saveData['TrainY'].to(device)
        TrainPrep = saveData['TrainPrep'].to(device)
        TestX_1 = saveData['TestX_1'].to(device)
        TestX_2 = saveData['TestX_2'].to(device)
        TestX_3 = saveData['TestX_3'].to(device)
        TestX_4 = saveData['TestX_4'].to(device)
        TestY = saveData['TestY'].to(device)
        TestPrep = saveData['TestPrep'].to(device)

        print('Load Ready!')

    selectMode = 2
    maxStep = 10000

    if (selectMode == 0): 
        optimizer = torch.optim.SGD(net.parameters(), lr=0.00001)
        loss_func = torch.nn.MSELoss()

        for t in range(maxStep):
            y = net(TrainX_1, TrainX_2, TrainX_3, TrainX_4)
            loss = loss_func(y.to(torch.float32), TrainY.to(torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if t % 100 == 0:
                print('Loss = %.4f (proc %.2f)' % (loss.data, t/maxStep))
        torch.save(net.state_dict(), './myModel.pth')
        print('Model Save!')

    elif (selectMode == 1):
        net.load_state_dict(torch.load("./myModel.pth"))

        optimizer = torch.optim.SGD(net.parameters(), lr=0.00001)
        loss_func = torch.nn.MSELoss()

        for t in range(maxStep):
            y = net(TrainX_1, TrainX_2, TrainX_3, TrainX_4)
            loss = loss_func(y.to(torch.float32), TrainY.to(torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if t % 100 == 0:
                print('Loss = %.8f (proc %.3f)' % (loss.data, t/maxStep))

        torch.save(net.state_dict(), './myModel.pth')
        print('Model Save!')

    else:
        net.load_state_dict(torch.load("./myModel.pth", map_location=torch.device(device)))
        net.eval()
        print('Model Load!')

    test = net(TestX_1, TestX_2, TestX_3, TestX_4)

    test = test * (TestPrep[:, 1].reshape(-1, 1) - TestPrep[:, 0].reshape(-1, 1)) + TestPrep[:, 0].reshape(-1, 1)
    testY = TestY * (TestPrep[:, 1].reshape(-1, 1) - TestPrep[:, 0].reshape(-1, 1)) + TestPrep[:, 0].reshape(-1, 1)

    print('Output \t Y \t delta \t deltaRate')

    print(torch.cat((test, testY, (test-testY), (test-testY)/testY), dim = 1))

    print(torch.std_mean((test-testY)/testY))