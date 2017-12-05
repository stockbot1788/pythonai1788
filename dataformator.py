import os
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from numpy import array

class DataFormator(object):
    def __init__(self, fullData):
        self.m_Data = fullData

    def formatData(self):
        inputX = []
        inputXR = []
        inputX2 = []
        inputX2R = []
        pos = []
        outputY = []
        longWin = 0
        shortWin = 0

        off = 0
        data1 = 0
        for singleDayData in self.m_Data:
            for i in range(35,len(singleDayData.m_data)-30):
                X1,X1r,X2,X2r,Pos = self.packDataForX2(singleDayData,i)
                #print(X1,X1r,X2,X2r,Pos)
                Yhat = self.packValueForY(singleDayData,i)
                data1 = data1 + 1
                if Yhat == 1:
                    off = off + 1
                outputY.append(Yhat)
                X1 = X1.reshape(30,5,1)
                inputX.append(X1)
                inputXR.append(X1r)     
                X2 = X2.reshape(30,4,1)           
                inputX2.append(X2)
                #print(X2.shape)
                inputX2R.append(X2r)
                pos.append(Pos)


        return inputX,inputXR,inputX2,inputX2R,pos,outputY

    def packDataForX2(self,singleDayData,idx):
        openNumber = np.float(singleDayData.m_data[0]._open)
        CandleArr = []
        PriceDiffArr = []
        MinMaxCandle = []
        MinMaxPriceDiff = []
        Position = []
        for i in range(idx-30,idx):
            crtData = singleDayData.m_data[i]
            pastData = singleDayData.m_data[i-1]

            _open = np.float(crtData._open)
            _close  = np.float(crtData._close)
            _high = np.float(crtData._high)
            _low  = np.float(crtData._low)

            df1 = _open -_close
            df2 = _open -_high
            df3 = _open -_low
            df4 = _close -_high
            df5 = _close -_low

            series = [df1,df2,df3,df4,df5]
            CandleArr.append(series)

            _openp = np.float(pastData._open)
            _closep  = np.float(pastData._close)
            _highp = np.float(pastData._high)
            _lowp  = np.float(pastData._low)

            dp1 = _open -_openp
            dp2 = _close -_closep
            dp3 = _high -_highp
            dp4 = _low -_lowp
            seriesp = [dp1,dp2,dp3,dp4]
            PriceDiffArr.append(seriesp)

            pos1 = openNumber - _open
            pos1 = pos1/400 * 100
            Position.append(pos1)
            
        CandleArr,MaxMinCandl = self.NormalizeMatrix(CandleArr)
        PriceDiffArr,MaxMinPrice = self.NormalizeMatrix(PriceDiffArr)
        MinMaxCandle = MaxMinCandl
        MinMaxPriceDiff = MaxMinPrice
        Position = array(Position)
        Position = [np.mean(Position, dtype=np.float64)]
        return CandleArr,MinMaxCandle,PriceDiffArr,MinMaxPriceDiff,Position

    def NormalizeMatrix(self,Matrix):
        ordm = [len(Matrix),len(Matrix[0])]
        Matrix = array(Matrix)
        Matrix  = Matrix.reshape(len(Matrix[0])*len(Matrix),1)
        series = Matrix
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(series)
        output = scaler.transform(series)
        output = output.reshape(ordm[0],ordm[1])
        min = np.float(scaler.data_min_)
        max = np.float(scaler.data_max_)
        MaxMin = [min,max]
        MaxMin = array(MaxMin)
        return output,MaxMin

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


        tmp = [df1,df2,df3,df4,df5]    
        return tmp 

    def packValueForY(self,singleDayData,step):
        longPrice = np.float(singleDayData.m_data[step]._high)
        sucess = 0
        
        for i in range(0,25):
            minData = singleDayData.m_data[step+i]
            earn = np.float(minData._high) - longPrice
            if earn > 35:
                sucess = 1
                break
        
        return sucess
        
        

            
