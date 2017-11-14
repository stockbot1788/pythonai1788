import gym
from gym import spaces
from DateStock import SingleDateStock
from DateStock import Stock
import sys
import codecs
from random import random
import numpy as np
import math
import os

class MarketEnv(gym.Env):
	def __init__(self, dir_path):
            self.data = []
            self.getEnvData(dir_path)
            self.actions = ["LONG","SHORT","WEAK"]
            self.action_space = spaces.Discrete(len(self.actions))
            self.observation_space = spaces.Box(np.ones(10)*-1,np.ones(10))
            self.biggestLost = -25
            self.superLoss = -50
            self.biggestEarn = 25
            self.superEarn = 50

    # self.longlist = []
    # self.shortlist = []
	def _step(self,action):
            self.reward = 0
            buyPrice = self.StockDataSingleDay.m_data[self.stepNumber]._high
            if self.actions[action] == "LONG":
               print("long")
            elif self.actions[action] == "SHORT":
               print("short")
            elif self.actions[action] == "WEAK":
               print("weak")

            self.defineState()
            self.stepNumber = self.stepNumber + 1
            if self.stepNumber > len(self.StockDataSingleDay.m_data)-1-15:
                self.done = True
            return self.state, self.reward, self.done, {}

	def _reset(self):
            self.cur_reward = 0
            self.reward = 0
            self.done = False
            self.stepNumber = 10
            randIdx = int(random() * len(self.data))
            self.StockDataSingleDay = self.data[randIdx]
            self.numberofdata = 10
            self.defineState()        
            return self.state

	def _seed(self):
            print("_seed")

	def defineState(self):
            
            tmpState = []
            _all=[]
            for i in range(self.numberofdata):
                _h = self.StockDataSingleDay.m_data[self.stepNumber-i]._high
                _l = self.StockDataSingleDay.m_data[self.stepNumber-i]._low
                _c = self.StockDataSingleDay.m_data[self.stepNumber-i]._close
                _o = self.StockDataSingleDay.m_data[self.stepNumber-i]._open
                tmp = [_h,_l,_c,_o]
                _all.append(tmp)
            _all = list(reversed(_all))

            Max = -99999999999
            Min = 999999999999
            for i in range(0,self.stepNumber):
                h = self.StockDataSingleDay.m_data[i]._high
                l = self.StockDataSingleDay.m_data[i]._low
                h = np.float(h)
                l = np.float(l)
                Max = max([h,Max])
                Min = min([l,Min])

            tmpState = _all
            X = np.array(tmpState)
            X = np.expand_dims(X, axis=0)
            tmpState = X
            tmpState = tmpState.reshape(-1)
            tmpState = tmpState.astype(np.float)
            #do normalization
            tmpState = [Max-np.float(i) for i in tmpState]
            tmpState.append(float(Max-Min))
            #tmpState = [float(i)/max(tmpState) for i in tmpState]
            tmpState.append(self.stepNumber/len(self.StockDataSingleDay.m_data))
            self.state = tmpState

	def getEnvData(self,path):
            filelist = os.listdir(path)
            datalist = []
            for val in filelist:
                if val.find(".txt") != -1:
                    path = "data/"+val
                    datalist.append(path)
            #datalist = datalist[:-6]
            i = 0
            for val in datalist:
                self.getSingleData(val)

	def getSingleData(self,path):
            f = codecs.open(path, "r", "utf-8")
            StockDataArr = []
            CrtDate = ""                
            for line in f:
                 if line.strip() != "":
                        dataDate = line.strip().split(",")[0].split(" ")[0]

                        if CrtDate!=dataDate:
                                t = SingleDateStock()
                                CrtDate = dataDate
                                StockDataArr.append(t)
                                StockDataArr[len(StockDataArr)-1].m_data = []
                                StockDataArr[len(StockDataArr)-1].m_date = ""+dataDate
                                CrtDate = dataDate

                        code = Stock()
                        code._open = line.strip().split(",")[1] #open
                        code._high = line.strip().split(",")[2] #high
                        code._low  = line.strip().split(",")[3] #low
                        code._close= line.strip().split(",")[4] #close
                        
                        StockDataArr[len(StockDataArr)-1].m_data.append(code)
            f.close()
            self.data = self.data + StockDataArr

            
