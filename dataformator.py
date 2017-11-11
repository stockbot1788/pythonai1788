import os
import numpy as np

class DataFormator(object):
    def __init__(self, fullData):
        self.m_Data = fullData

    def formatData(self):
        inputX = []
        outputY = []
        for singleDayData in self.m_Data:
            #singleDayData.m_data = singleDayData.m_data[:-20]
            #singleDayData.m_data = singleDayData.m_data[15:]
            for step in range(15,len(singleDayData.m_data)-20):
                singleDataX = []
                #pack for X
                for i in range(0,10):
                    data = self.packDataForX(singleDayData,step-i)
                    singleDataX.append(data)
                singleDataX = np.array(singleDataX)
                singleDataX = singleDataX.reshape(-1)
                #print("----")
                #print(singleDataX.shape)
                #singleDataX = np.expand_dims(singleDataX, axis=2) 
                #print(singleDataX.shape)
                #print("----")
                inputX.append(singleDataX)
                y = self.packValueForY(singleDayData,step)
                outputY.append(y)
        return inputX,outputY

    def packDataForX(self,singleDayData,idx):
        minData = singleDayData.m_data[idx]
        minDataB4 = singleDayData.m_data[idx-1]
        _open = np.float(minData._open)
        _close  = np.float(minData._close)
        _high = np.float(minData._high)
        _low  = np.float(minData._low)

        _openb4 = np.float(minDataB4._open)
        _closeb4  = np.float(minDataB4._close)
        _highb4 = np.float(minDataB4._high)
        _lowb4  = np.float(minDataB4._low)

        df1 = _open -_close
        df2 = _open -_high
        df3 = _open -_low
        df4 = _close -_high
        df5 = _close -_low

        df6 = _open - _openb4
        df7 = _close - _closeb4
        df8 = _high - _highb4
        df9 = _low -_lowb4

        tmp = [df1,df2,df3,df4,df5,df6,df7,df8,df9]    
        return tmp 

    def packValueForY(self,singleDayData,step):
        longPrice = np.float(singleDayData.m_data[step]._high)
        status = 0
        for i in range(0,15):
            minData = singleDayData.m_data[step+i]
            loss = longPrice - np.float(minData._low)
            earn = np.float(minData._high) - longPrice
            if loss > 25:
                status = 1
                break
            if earn > 25:
                status = 2
                break
        if status != 2:
           status = 0
           shortPrice = np.float(singleDayData.m_data[step]._low)
           for i in range(0,15): 
               minData = singleDayData.m_data[step+i]
               loss = np.float(minData._high) - shortPrice
               earn = shortPrice - np.float(minData._low)
               if loss > 25:
                status = 1
                break
               if earn > 25:
                status = 3
                break
        if status == 0:
            status = 1
        status = status - 1
        return status
        
        

            
