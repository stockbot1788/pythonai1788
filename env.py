import gym
from gym import spaces
from DateStock import SingleDateStock
from DateStock import Stock
import sys
import codecs
from random import random
import numpy as np
import math

class MarketEnv(gym.Env):
	def __init__(self, dir_path):
            self.getEnvData(dir_path)
            self.actions = ["LONG","SHORT","SELL_L","SELL_S","IGNORE"]
            self.action_space = spaces.Discrete(len(self.actions))
            self.observation_space = spaces.Box(np.ones(10)*-1,np.ones(10))

    # self.longlist = []
    # self.shortlist = []
	def _step(self,action):
            self.reward = 0
            if self.done:
                return self.state, self.reward, self.done, {}

            if self.actions[action] == "LONG":
                if len(self.longlist) == 1:
                    self.reward = -300
                else:
                    self.longlist.append(self.StockDataSingleDay.m_data[self.stepNumber]._high)
            elif self.actions[action] == "SHORT":
                if len(self.shortlist) == 1:
                    self.reward = -300
                else:
                    self.shortlist.append(self.StockDataSingleDay.m_data[self.stepNumber]._high)
            elif self.actions[action] == "SELL_L":
                if len(self.longlist) == 0:
                    self.reward = -300
                else:
                   self.reward = int(self.StockDataSingleDay.m_data[self.stepNumber]._high) - int(self.longlist[0])
                   self.longlist = []
            elif self.actions[action] == "SELL_S":
                if len(self.shortlist) == 0:
                    self.reward = -300
                else:
                    self.reward = int(self.shortlist[0]) - int(self.StockDataSingleDay.m_data[self.stepNumber]._high)
                    self.shortlist = []
            elif self.actions[action] == "IGNORE":
                 self.reward = 0

            self.defineState()
            self.stepNumber = self.stepNumber + 1
            if self.stepNumber >= len(self.StockDataSingleDay.m_data):
                self.done = True
            if self.reward < -100:
                self.done = True
            # if self.reward > 200:
            #     self.done = True
            return self.state, self.reward, self.done, {}

	def _reset(self):
            self.reward = 0
            self.done = False
            self.longlist = []
            self.shortlist = []
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

            tmpState = _all
            #print(tmpState)
            X = np.array(tmpState)
            X = np.expand_dims(X, axis=0)
            tmpState = X
            tmpState = tmpState.reshape(-1)
            #norm = [np.float(i)/max(tmpState) for i in tmpState]
            tmpState = tmpState.astype(np.float)
            tmpState = [np.float(i)/max(tmpState) for i in tmpState]
       
            tmpState = np.concatenate((tmpState,[len(self.longlist)]),axis=0)
            tmpState = np.concatenate((tmpState,[len(self.shortlist)]),axis=0)
            #tmpState = [len(self.longlist),len(self.shortlist)]
            #print("data {}",format(tmpState))
            self.state = tmpState

    #self.data containlist of arr
	def getEnvData(self,path):
            f = codecs.open("data/20150917.txt", "r", "utf-8")
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
                        code._open = line.strip().split(",")[1]
                        code._high = line.strip().split(",")[1]
                        code._low  = line.strip().split(",")[1]
                        code._close= line.strip().split(",")[1]
                        StockDataArr[len(StockDataArr)-1].m_data.append(code)
            f.close()
            self.data = StockDataArr
