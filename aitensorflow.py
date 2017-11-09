import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
from env import MarketEnv

env = MarketEnv("data")

inputX = []
outputY = []
for data in env.data:
    for i in range(20,len(data.m_data)-1-15):
        _all=[]
        for j in range(i-10,i):
            _h = data.m_data[j]._high
            _l = data.m_data[j]._low
            _c = data.m_data[j]._close
            _o = data.m_data[j]._open
            tmp = [_h,_l,_c,_o]
            _all.append(tmp)
        _all = list(reversed(_all))

        Max = -99999999999
        Min = 999999999999
        for j in range(0,i):
            h = data.m_data[j]._high
            l = data.m_data[j]._low
            h = np.float(h)
            l = np.float(l)
            Max = max([h,Max])
            Min = min([l,Min])                    
        X = np.array(_all)
        X = np.expand_dims(X, axis=0)
        X = X.reshape(-1)
        X = X.astype(np.float)
        X = [Max-np.float(i) for i in X]
        X.append(Max-Min)
        X = [100*i/max(X) for i in X]
        X.append(i)
        inputX.append(X)

        #work for Y value 
        LongPrice = data.m_data[i]._high
        LongPrice = np.float(LongPrice)
        ShortPrice = data.m_data[i]._low
        MaxEarn = -99999
        MaxLoss = 99999
        for j in range(1,15):
            tmpPriceHigh = data.m_data[i+j]._high
            tmpPriceHigh = np.float(tmpPriceHigh)
            tmpPriceLow = data.m_data[i+j]._low
            tmpPriceLow = np.float(tmpPriceLow)
            
            earnShort = ShortPrice - tmpPriceLow
            lossShort = ShortPrice - tmpPriceHigh

            earnLong = tmpPriceHigh - LongPrice
            lossLong = tmpPriceLow - LongPrice

            MaxLoss = min([lossLong,lossShort,MaxLoss])
            MaxEarn = max([earnLong,earnShort,MaxEarn])

        