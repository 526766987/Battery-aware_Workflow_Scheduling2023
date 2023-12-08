import torch
from taskGenerator import taskGenerator
from ALG_HEFT import HEFT
from ALG_DC_MBFLS import DC_MBFLS
from DNN_Apply import DNN_Net_Solution
from ALG_DC_MBFLS_DNNkLP import DC_MBFLS_DNNkLP
from ALG_MILP import MILP
from scipy.io import savemat
import datetime
import numpy as np
import math

workflow = taskGenerator(5, 3)

ALG_Enable = [1,0,1,1]

printFlag = True

if ALG_Enable[0]:
	HEFT_Exm = HEFT(workflow.N, workflow.P, workflow.c, workflow.w)
	if printFlag:
		print("HEFT use time: {0}".format(HEFT_Exm.useTime))
		print(HEFT_Exm.taskRecord)

if ALG_Enable[1]:
	DC_MBFLS_Exm = DC_MBFLS(workflow.N, workflow.P, workflow.c, workflow.w, workflow.price, workflow.ddl)
	if printFlag:
		print("DC_MBFLS use time: {0}".format(DC_MBFLS_Exm.useTime))
		print(DC_MBFLS_Exm.stage2_taskRecord)

if ALG_Enable[2]:
	net = DNN_Net_Solution()
	DC_MBFLS_DNNkLP_Exm = DC_MBFLS_DNNkLP(workflow.N, workflow.P, workflow.c, workflow.w, workflow.price, workflow.ddl, net)
	if printFlag:
		print("DC_MBFLS_DNNkLP use time: {0}".format(DC_MBFLS_DNNkLP_Exm.useTime))
		print(DC_MBFLS_DNNkLP_Exm.stage2_taskRecord)

if ALG_Enable[3]:
	MILP_Exm = MILP(workflow.N, workflow.P, workflow.c, workflow.w, workflow.ddl, workflow.price, 600)
	if printFlag:
		print("MILP use time: {0}".format(MILP_Exm.useTime))
		print("MILP solve time: {0}".format(MILP_Exm.solveTime))
		print(MILP_Exm.taskRecord)