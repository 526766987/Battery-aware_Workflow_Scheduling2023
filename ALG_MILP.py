import docplex.mp.model as cpx
import cplex
import torch
import math
import time

class MILP(object):
	def __init__(self, N, P, c, w, ddl, price, timelimit):
		time_start = time.clock()

		self.N = N
		self.P = P
		self.c = c
		self.w = w
		self.ddl = ddl
		self.price = price

		opt_model = cpx.Model(name="MIP Model")
		opt_model.context.cplex_parameters.emphasis.mip = 1
		opt_model.context.cplex_parameters.timelimit = timelimit

		x = {(i, k, t): opt_model.binary_var(name="{0}_{1}_{2}".format(i, k, t))\
			for i in range(0, self.N)\
			for k in range(0, self.P)\
			for t in range(0, math.floor(self.ddl) + 1)\
			}

		opt_model.add_constraints(\
			cts = [(opt_model.sum(x[(i, k, t)] for t in range(0, math.floor(self.ddl) + 1) for k in range(0, P)) == 1)\
			for i in range(0, self.N)]\
			)

		opt_model.add_constraint(\
			ct = (opt_model.sum(x[(0, k, 0)] for k in range(0, self.P)) == 1)\
			)

		opt_model.add_constraint(\
			ct = (opt_model.sum(\
				(t + math.ceil(self.w[self.N - 1, k].item())) * x[(self.N - 1, k, t)]\
				for k in range(0, self.P)\
				for t in range(0, math.floor(self.ddl) + 1))\
				<= self.ddl)\
			)

		for i in range(0, self.N):
			for j in range(0, self.N):
				if (self.c[i, j] >= 0):
					opt_model.add_constraint(\
						ct = (opt_model.sum(t * x[(j, k, t)] - (t + math.ceil(self.w[i, k].item())) * x[(i, k, t)]\
							for k in range(0, self.P)\
							for t in range(0, math.floor(self.ddl) + 1))\
							- (1-opt_model.sum(opt_model.sum(x[(i,k,t)] for t in range(0, math.floor(self.ddl) + 1)) * opt_model.sum(x[(j,k,t)] for t in range(0, math.floor(self.ddl) + 1)) for k in range(0, self.P))) * math.ceil(self.c[i, j].item()))>= 0\
						)

		opt_model.add_constraints(\
			cts = [(opt_model.sum(x[(i, k, t)] for t in range(max(0, tc - math.ceil(self.w[i, k].item()) + 1), tc + 1) for i in range(0, self.N)) <= 1)\
			for k in range(0, self.P)\
			for tc in range(0, math.floor(self.ddl) + 1)]\
			)

		objective = opt_model.minimize(opt_model.sum_squares(\
			opt_model.sum(\
			x[(i, k, t)] * self.price[k, 0].item()\
			for k in range(0, self.P)\
			for i in range(0, self.N)\
			for t in range(max(0, tc - math.ceil(self.w[i, k].item()) + 1), tc + 1))\
			for tc in range(0, math.floor(self.ddl) + 1)\
			))

		time_solve = time.clock()

		solution = opt_model.solve(agent='local')

		self.taskRecord = torch.zeros(self.N, 3)
		self.processRecord = [torch.tensor([[]]) for i in range(0, self.P)]
		processRecordCount = torch.zeros(self.P)

		#print(solution)

		# output
		if (solution is not None):
			for val in solution.iter_var_values():
				info = (str(val[0])).split('_')
				i = int(info[0])
				k = int(info[1])
				t = int(info[2])
				self.taskRecord[i, 0] = t
				self.taskRecord[i, 1] = t + self.w[i, k]
				self.taskRecord[i, 2] = k

				if (processRecordCount[k].item() == 0):
					self.processRecord[k] = torch.tensor([[t, t + self.w[i, k], i]])
				else:
					self.processRecord[k] = torch.cat((self.processRecord[k], torch.tensor([[t, t + self.w[i, k], i]])), dim = 0)
				processRecordCount[k] = processRecordCount[k] + 1

		time_end = time.clock()
		self.useTime = time_end - time_start
		self.solveTime = time_end - time_solve