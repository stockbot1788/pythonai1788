import os
import numpy as np

class DataFormator(object):
    def __init__(self, fullData):
        self.m_Data = fullData

    def formatData(self):
        inputX = []
        inputX2 = []
        outputY = []
        longWin = 0
        shortWin = 0

        for singleDayData in self.m_Data:
            #singleDayData.m_data = singleDayData.m_data[:-20]
            #singleDayData.m_data = singleDayData.m_data[15:]
            for step in range(15,len(singleDayData.m_data)-20):
                singleDataX = []
                singleDataX2 = []
                #pack for X
                for i in range(0,10):
                    data = self.packDataForX(singleDayData,step-i)
                    data2 = self.packDataForX2(singleDayData,step-i)
                    singleDataX.append(data)
                    singleDataX2.append(data2)
                singleDataX = np.array(singleDataX)
                #singleDataX = singleDataX.reshape(-1)
                #print("----")
                #print(singleDataX.shape)
                singleDataX = np.expand_dims(singleDataX, axis=2) 
                #print(singleDataX.shape)
                #print("----")
                singleDataX2 = np.array(singleDataX2)
                singleDataX2= np.expand_dims(singleDataX2, axis=2) 
                #print(singleDataX2.shape)
                y = self.packValueForY(singleDayData,step)
                
                #temp = [singleDataX2,singleDataX]
                #temp = [np.array(i) for i in temp]
                
                inputX.append(singleDataX)
                inputX2.append(singleDataX2)
                outputY.append(y)
                if y == 0:
                    longWin = longWin + 1
                elif y == 2:
                    shortWin = shortWin + 1
        inputX = [np.array(inputX2),np.array(inputX)]
        print(shortWin,longWin,len(outputY))
        return inputX,outputY

    def packDataForX2(self,singleDayData,idx):
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

        # df1 = _open -_close
        # df2 = _open -_high
        # df3 = _open -_low
        # df4 = _close -_high
        # df5 = _close -_low
        
        df6 = _open - _openb4
        df7 = _close - _closeb4
        df8 = _high - _highb4
        df9 = _low -_lowb4

        tmp = [df6,df7,df8,df9]    
        return tmp 

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
        
        #df6 = _open - _openb4
        #df7 = _close - _closeb4
        #df8 = _high - _highb4
        #df9 = _low -_lowb4

        tmp = [df1,df2,df3,df4,df5]    
        return tmp 

    def packValueForY(self,singleDayData,step):
        longPrice = np.float(singleDayData.m_data[step]._high)
        longStatus = 0
        
        for i in range(0,15):
            minData = singleDayData.m_data[step+i]
            loss = longPrice - np.float(minData._low)
            earn = np.float(minData._high) - longPrice
            if loss > 25:
                longStatus = 1
                break
            if earn > 25:
                longStatus = 2
                break
        shortStatus = 0
        shortPrice = np.float(singleDayData.m_data[step]._low)
        for i in range(0,15): 
            minData = singleDayData.m_data[step+i]
            loss = np.float(minData._high) - shortPrice
            earn = shortPrice - np.float(minData._low)
            if loss > 25:
                shortStatus = 1
                break
            if earn > 25:
                shortStatus = 2
                break
        status = 1
        if shortStatus == 2:
            status = 2
        elif longStatus == 2:
            status = 0

        #status = status - 1
        return status
        
        

            
