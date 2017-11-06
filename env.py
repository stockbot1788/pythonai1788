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
            print("_init")
            self.getEnvData(dir_path)
            self.actions = ["LONG","SHORT","SELL_L","SELL_S"]
            self.action_space = spaces.Discrete(len(self.actions))
            self.observation_space = spaces.Box(np.ones(10)*-1,np.ones(10))

    # self.longlist = []
    # self.shortlist = []
	def _step(self,action):
            ##print(self.StockDataSingleDay.m_data[self.stepNumber]._high)
            #print(len(self.StockDataSingleDay.m_data))
            if self.done:
                return self.state, self.reward, self.done, {}
            self.reward = 0
            _reward = 0
            if self.actions[action] == "LONG":
                #check if long list contain stock return -1
                #if not append to long list 
                if len(self.longlist) == 1:
                    _reward = -999
                else:
                    self.longlist.append(self.StockDataSingleDay.m_data[self.stepNumber]._high)
                    #print("add longlist")
            elif self.actions[action] == "SHORT":
                #check if short list contain stock return -1
                #if not append to short list 
                if len(self.shortlist) == 1:
                    _reward = -999
                else:
                    self.shortlist.append(self.StockDataSingleDay.m_data[self.stepNumber]._high)
            elif self.actions[action] == "IGNORE":
                _reward = -5    
            elif self.actions[action] == "SELL_L":
                #print("sell long")
                if len(self.longlist) == 0:
                    _reward = -1200
                else:
                   _reward = int(self.StockDataSingleDay.m_data[self.stepNumber]._high) - int(self.longlist[0])
                   self.longlist = []
                   print("minor {}".format(_reward))
            elif self.actions[action] == "SELL_S":
                #print("sell short")
                #print(len(self.shortlist))
                if len(self.shortlist) == 0:
                    _reward = -1200
                else:
                    _reward = int(self.shortlist[0]) - int(self.StockDataSingleDay.m_data[self.stepNumber]._high)
                    self.shortlist = []
                    #print("-----")
                    #print(_reward)
                #check if short list is empty return -1
                #use short list data and calculate earn or loss

            
            if _reward < -200:
                self.reward = -100
            else:
                self.reward = self.reward + _reward
            self.stepNumber = self.stepNumber + 1
            if self.stepNumber >= len(self.StockDataSingleDay.m_data):
                self.done = True
            if self.reward < -50:
                self.done = True
            if self.reward > 105:
                self.done
            return self.state, self.reward, self.done, {}

	def _reset(self):
            self.reward = 0
            self.done = False
            self.longlist = []
            self.shortlist = []
            self.stepNumber = 0
            randIdx = int(random() * len(self.data))
            #print(randIdx)
            self.StockDataSingleDay = self.data[randIdx]
            self.currentTargetIndex = 10
            self.numberofdata = 10
            self.defineState()        
            return self.state

	def _seed(self):
            print("_seed")

	def defineState(self):
            
            tmpState = []
            _all=[]
            for i in range(self.numberofdata):
                _h = self.StockDataSingleDay.m_data[self.currentTargetIndex-i]._high
                _l = self.StockDataSingleDay.m_data[self.currentTargetIndex-i]._low
                _c = self.StockDataSingleDay.m_data[self.currentTargetIndex-i]._close
                _o = self.StockDataSingleDay.m_data[self.currentTargetIndex-i]._open
                tmp = [_h,_l,_c,_o]
                _all.append(tmp)
            _all = list(reversed(_all))

            tmpState = _all
            X = np.array(tmpState)
            X = np.expand_dims(X, axis=0)
            tmpState = X
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
