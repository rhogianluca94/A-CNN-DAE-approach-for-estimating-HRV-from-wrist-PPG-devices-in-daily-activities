import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

class PRVtracking:

    def __init__(self, aHR, bHR, aSTD, bSTD, aRMSSD, bRMSSD, theta):
        self.aHR = aHR
        self.bHR = bHR
        self.aSTD = aSTD
        self.bSTD = bSTD
        self.aRMSSD = aRMSSD
        self.bRMSSD = bRMSSD
        self.theta = theta


    def segmentRR(self, y, srate):
        rrOut = []
        M = len(y)
        winStep = 8 * srate
        nn = 0

        for ii in range(0, M, winStep):
            winStart = ii
            winStop = winStart + winStep

            if winStop > M:
                break

            rrOut.append(y[winStart:winStop])
            nn += 1

        return np.array(rrOut)

    def segment_rr_intervals(self, rr):
        time = (np.cumsum(rr) - rr[0])
        L = time[-1]   
        wlen = 10       
        e = wlen#8
        ii = 0
        rr_segments = []

        while e < L:
            if ii == 0:
                low = 0  
            else:
                low = np.where(time > s)[0][0]  
            high = np.where(time < e)[0][-1]  

            rr_segments.append(rr[low:high+1])  
            ii += 1
            s = e
            e = s + wlen
        return rr_segments

    def LRprediction_win(self, values, x, xpred, weights):
        y = np.array(values).reshape(-1, 1)
        x = np.array(x).reshape(-1, 1)
        xpred = np.array(xpred).reshape(1, -1)
        weights = np.array(weights)

        # Create and fit the linear regression model with weighted samples
        model = LinearRegression()
        model.fit(x, y, sample_weight=weights) 

        # Predict the value for xpred
        hpred = model.predict(xpred)

        return hpred[0]


    def estimateMeasure(self, rr, measureType, operator):
        match measureType:
            case 'HR':
                if operator=='mean':
                    return 60 / np.mean(rr)
                elif operator=='median':
                    return 60 / np.median(rr)
            case 'stdRR':
                if operator=='mean':
                    return np.std(rr) * 1e3
                elif operator=='median':
                    return np.sqrt( 1/(len(rr)-1) * np.sum( [(rri-np.median(rr))**2 for rri in rr] ) ) * 1e3
            case 'RMSSD':
                if operator=='mean':
                    return np.sqrt(np.mean(np.diff(rr)**2)) * 1e3
                elif operator=='median':
                    return np.sqrt(np.median(np.diff(rr)**2)) * 1e3
            case _:
                raise ValueError('{} is not a valid measure'.format(str(measureType)))
                

    def tracking(self, rr, a, b, theta, measureType):
        N = len(rr)  # nWindows
        ecount = 0
        Th0 = a  # (bpm) D|HR| between hi and hi-1 cannot be greater than that
        maxErr = theta
        #rr = 60.0 / rr  # Convert RR intervals to heart rate (bpm)
        
        # Estimate h1
        hrec = np.array([self.estimateMeasure(rr_segment, measureType, operator='mean') for rr_segment in rr])
        #hrec = np.mean(rr, axis=1)
        winSize = 10
        past = 0.75  # e.g., 1.5 / 0.5 : (1+past) / (1-past)

        for kk in range(3):
            ii = 2  # Start from the 3rd window
            while ii < N:
                if kk == 0:
                    hi = self.estimateMeasure(rr[ii], measureType, operator='mean')
                    #hi = np.mean(rr[ii, :])  # Estimate the mean HR in the i-th window
                else:
                    hi = hrec[ii]
                
                hiprev = hrec[ii - 1]  # HR of the previous window
                Dh = hi - hiprev
                The = Th0 + b * ecount

                if abs(Dh) > The:  # The variation exceeds the threshold
                    hi = self.estimateMeasure(rr[ii], measureType, operator='median')
                    if abs(hi - hiprev) < The:
                        hrec[ii] = hi
                        ii += 1
                        continue
                    else:
                        ecount += 1
                        if ecount >= maxErr:  # Prediction unreliable, clip the error
                            hi = hiprev + np.sign(Dh) * min(Th0, abs(Dh))
                        else:
                            # Predict hi using linear regression (LR) with observations around it
                            if ii - round(winSize / 2 - 1) <= 0:
                                ecc = -1 * (ii - round(winSize / 2 - 1))
                                X = np.arange(ii - round(winSize / 2 - 1) + ecc + 1, ii + round(winSize / 2 - 1) - ecc)
                            elif ii + round(winSize / 2 - 1) <= N:
                                X = np.arange(ii - round(winSize / 2 - 1), ii + round(winSize / 2 - 1) + 1)
                            else:
                                ecc = ii + round(winSize / 2 - 1) - N
                                X = np.arange(ii - round(winSize / 2 - 1) - ecc, N)

                            hLR = hrec[X-1] # occhio: python parte da 0! 
                            weights = np.ones(len(hLR))
                            weights[:np.where(X == ii)[0][0]] = (1 + past)
                            weights[np.where(X == ii)[0][0] + 1:] = (1 - past)
                            hLR = np.delete(hLR, np.where(X == ii))
                            weights = np.delete(weights, np.where(X == ii))
                            X = np.delete(X, np.where(X == ii))
                            
                            # Use weighted linear regression for prediction
                            hipred = self.LRprediction_win(hLR, X, ii, weights)
                            hrec[ii] = hipred
                            ii += 1
                            continue

                ecount = 0
                hrec[ii] = hi
                ii += 1

        return hrec


    def __call__(self, rrData, plotData=0):

        # Segment RR data
        rrData_seg = self.segment_rr_intervals(rrData)

        # RR parameters tracking
        hrTracked = self.tracking(rrData_seg, self.aHR, self.bHR, self.theta, measureType='HR')
        stdRRTracked = self.tracking(rrData_seg, self.aSTD, self.bSTD, self.theta, measureType='stdRR')
        rmssdTracked = self.tracking(rrData_seg, self.aRMSSD, self.bRMSSD, self.theta, measureType='RMSSD')
        
        hrOrig = np.array([self.estimateMeasure(rr, measureType='HR', operator='mean') for rr in rrData_seg])
        stdRROrig = np.array([self.estimateMeasure(rr, measureType='stdRR', operator='mean') for rr in rrData_seg])
        rmssdOrig = np.array([self.estimateMeasure(rr, measureType='RMSSD', operator='mean') for rr in rrData_seg])
        
        if plotData:
            plt.figure()
            plt.title('mean HR')
            plt.plot(hrOrig, linewidth=2, label='Before Tracking')
            plt.plot(hrTracked, linewidth=2, label='After Tracking')
            plt.legend()
            
            plt.figure()
            plt.title('std RR')
            plt.plot(stdRROrig, linewidth=2, label='Before Tracking')
            plt.plot(stdRRTracked, linewidth=2, label='After Tracking')
            plt.legend()

            plt.figure()
            plt.title('RMSSD')
            plt.plot(rmssdOrig, linewidth=2, label='Before Tracking')
            plt.plot(rmssdTracked, linewidth=2, label='After Tracking')
            plt.legend()
            plt.show()
           
        return hrTracked, stdRRTracked, rmssdTracked

