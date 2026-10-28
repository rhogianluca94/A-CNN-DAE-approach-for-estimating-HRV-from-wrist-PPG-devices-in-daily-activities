import os
from pathlib import Path
import numpy as np
from scipy import signal, interpolate
import torch    
from CNN_IEEESPCData import DAE_CNN
from trackingAlgorithm import PRVtracking
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
torch.set_warn_always(False)
class PPGdenoising:

    def __init__(self, srateOrig):
        self.filesep = os.path.sep
        self.kernelSize = 32
        self.srateOrig = srateOrig
        self.srate = 125.0
        self.srateRR = 4
        self.sequence_length = 8 * int(self.srate)

        # base folder
        scriptPath = os.path.dirname(__file__)
        # models directory
        self.modelsDir_DALIA = os.path.join(scriptPath, 'models') + ('\\model__LOO' + '1' + '_bestModel')

    @staticmethod
    def get_windows(data, Mt, olap):
        # Split a signal of length n into olap% overlapping windows each containing Mt terms 
        n = len(data)
        data_new = []
        ist = 0
        win = 0
        while 1:
            ied = ist + Mt
            if ied > n:
                break
            data_new.append(data[ist:ied])
            ist += int(Mt * (1 - olap/100))
        data_new_vec = np.array(data_new)   
        return data_new_vec

    @staticmethod
    def ppg_filter(ppg, srate):
        fs = srate
        fL = 0.7
        fH = 8
        filter_order = 8
        sos = signal.butter(filter_order, [fL, fH], btype='bandpass', output='sos', fs=fs)
        filtered_ppg = signal.sosfilt(sos, ppg) * -1
        return filtered_ppg
    
    def normalize(data):
        N = np.shape(data)[0]
        for kk in range(0, N):
            tmp = data[kk,:]
            tmp = (tmp-tmp.min())/(tmp.max()-tmp.min())
            data[kk,:] = tmp
        return data

    def prepData(self, data_ppg, winLen, overlap):
        
        """
        Preprocesses the PPG signal:
        - Resamples the signal to 125 Hz if the original sampling rate differs.
        - Filters the signal.
        - Segments the signal into overlapping windows of specified length.
        - Normalizes each segment to the range [0, 1].

        Parameters
        ----------
        data_ppg : np array
            Raw PPG signal to be preprocessed.
        winLen : float
            Length of each window in seconds.
            Use default value of 8sec.
        overlap : float
            Fractional overlap between consecutive windows (e.g., 0.5 for 50% overlap). 
            Use default value of 0.

        Returns
        -------
        data_ppg : ndarray
            A 2D array where each row corresponds to a PPG window.
        """
        
        if self.srateOrig != 125:
            print("Resampling at 125 Hz...")
            tOriginal = np.arange(0, data_ppg.shape[0]/self.srateOrig, 1/self.srateOrig)
            minLen = min(np.shape(tOriginal)[0], np.shape(data_ppg)[0])
            tOriginal = tOriginal[0:minLen]
            data_ppg = data_ppg[0:minLen]
            interpolator = interpolate.interp1d(tOriginal, data_ppg.squeeze(), kind='cubic')
            tResampled = np.arange(0, data_ppg.shape[0]/self.srateOrig, 1/self.srate)
            # handle the case where the target time axis is longer due to non integer ratio between new/old srates
            if tResampled[-1] > tOriginal[-1]:
                tResampled = tResampled[tResampled <= tOriginal[-1]]
            data_ppg = interpolator(tResampled)
        #filtering
        data_ppg = PPGdenoising.ppg_filter(ppg=data_ppg, srate=self.srate)
        print("Segmenting the PPG time series into {}s-long windows ...".format(winLen))
        data_ppg = np.squeeze(PPGdenoising.get_windows(data_ppg, int(winLen*self.srate), olap=overlap))
        # normalize between [0, 1]
        data_ppg = PPGdenoising.normalize(data_ppg)
        return data_ppg

    def runDenoisingModel(self, ppg_data, nModels=10):
        
        """
        Applies an ensemble of denoising autoencoder (DAE) CNN models to clean PPG data.

        This method loads a set of pre-trained convolutional autoencoder models and applies
        them to the input segmented PPG signal. It aggregates the output of multiple models
        to produce an ensemble denoised signal.

        Parameters
        ----------
        ppg_data : ndarray
            2D array of segmented PPG signals of shape (n_windows, n_samples).
        nModels : int, optional
            Number of DAE models to use for the ensemble. Default is 10.

        Returns
        -------
        x_out_ensemble : ndarray
            2D array containing the ensemble denoised PPG signal, with shape (n_windows * n_samples, 1).
        x_in_resh : ndarray
            1D array of the original input signal, reshaped to match the output format.

        Notes
        -----
        - This method assumes that `ppg_data` has already been preprocessed and windowed.
        
        See Also
        --------
        DAE_CNN : Denoising autoencoder model architecture used for inference.
        """
        
        print('Denoising PPG data ...')
        # Model construction
        [Nwin, Ns] = np.shape(ppg_data)
        nModels = 10
        model = DAE_CNN(kernel_size=self.kernelSize, dropoutRate=0.0)
        x_out_resh_all = np.zeros((Nwin*Ns, nModels)) # length input ppg x Nmodels
        for modelIndex in range(0,nModels):
            # input data
            x_val = torch.from_numpy(ppg_data)
            x_val = x_val.unsqueeze(1).float()

            # Load the saved model weights
            modelFilename = Path(str(self.modelsDir_DALIA) +  f'_M{modelIndex+1}.h5')
            model.load_state_dict(torch.load(modelFilename, map_location='cpu', weights_only=True))
            model.eval()  # Set the model to evaluation mode

            # Make predictions
            with torch.no_grad():
                outputs = model(x_val)

            x_out = outputs.numpy().squeeze()
            x_val = x_val.numpy().squeeze()

            # concatenate the data windows together
            x_out_resh = PPGdenoising.reshape_outputData_model(x_out)
            x_out_resh_all[:, modelIndex] = x_out_resh
            x_in_resh = PPGdenoising.reshape_outputData_model(x_val)

        # errors correlation matrix and weights calculation
        if(nModels>1):
            ensembleWeights = np.ones(shape=(nModels,))/nModels
            x_out_ensemble = np.sum(x_out_resh_all * ensembleWeights.T, axis=1, keepdims=True)
        else:
            x_out_ensemble = x_out_resh_all
        return x_out_ensemble, x_in_resh  

    def getPRVfeatures(self, ppg_data, step = None):
        
        """
        Computes the mean HR, stdRR, and the RMSSD from PPG 

        Parameters
        ----------
        ppg_data : np array
            The input PPG signal.
        step : int, optional
            The window size in seconds for segmenting the signal. If None,
            it defaults to the minimum between the data length and 30 minutes.

        Returns
        -------
        hr : array
            The estimated heart rate values for each window.
        stdrr : array
            The standard deviation of RR intervals for each window.
        rmssd : array
            The RMSSD values for each window.
        """
        
        print('Extracting PRV and PRV time-domain parameters ...')
        
        if step == None:
            step = min(np.shape(ppg_data)[0], int(30*60*self.srate))
        elif step < 30:
            print('You cannot specify a granularity lower than 30s')
            return None, None, None
        else:
            step = int(step * self.srate)
        
        # initialize HRV parameters tracking 
        prvTracking = PRVtracking(aHR=4, bHR=8, aSTD=16, bSTD=20, aRMSSD=5, bRMSSD=16, theta=3)

        # extract PRV from denoised PPG
        lenPPG = np.shape(ppg_data)[0]
        hr_vec, stdrr_vec, rmssd_vec = np.array([]), np.array([]), np.array([])
        for ii in range(0, lenPPG, step):    
            wStart = ii
            wStop = wStart + step 
        
            x = ppg_data[wStart:wStop]      
            peaks_x_,_, peaks = PPGdenoising.detect_peaks_msptd_V2(torch.tensor(x).squeeze(1), srate=self.srate)
            #peaks_x_,_ = PPGdenoising.detect_peaks_msptd_V2(torch.tensor(x), srate=self.srate)
            outRR = peaks_x_[1:].numpy()

            # perform PRV parameters tracking
            hr, stdRR, rmssd = prvTracking(rrData=outRR, plotData = 0)
            hr_vec = np.concatenate([hr_vec, hr])
            stdrr_vec = np.concatenate([stdrr_vec, stdRR])
            rmssd_vec = np.concatenate([rmssd_vec, rmssd])
        return hr_vec, stdrr_vec, rmssd_vec


    def detect_peaks_msptd_V2(x, srate):
            
        N = len(x)  # length of signal
        L = int(np.ceil(N / 512)) - 1  # max window length

        # - initialise LMS matrices
        m_max = torch.zeros((int(L), N), dtype=torch.bool) 
        m_min = torch.zeros((int(L), N), dtype=torch.bool) 

        # - populate LMS matric
        k_values = torch.arange(1, L + 1)
        i_range = torch.arange(k_values.min() + 2, N - k_values.max() + 1)
        # Condition for local maxima
        m_max[:, i_range - 1] = (x[i_range - 1] > x[i_range - k_values[:, None] - 1]) & (x[i_range - 1] > x[i_range + k_values[:, None] - 1]) 
        # Condition for local minima
        m_min[:, i_range - 1] = (x[i_range - 1] < x[i_range - k_values[:, None] - 1]) & (x[i_range - 1] < x[i_range + k_values[:, None] - 1]) 
        
        # find the scale with the most local maxima (or local minima)
        # - row-wise summation (i.e. sum each row)
        gamma_max = torch.sum(m_max, dim=1)  # row-wise
        gamma_min = torch.sum(m_min, dim=1)  # row-wise
        # - find scale with the most local maxima (or local minima)
        _, idxs_max = torch.max(gamma_max, dim=0)
        _, idxs_min = torch.max(gamma_min, dim=0)

        #  remove all elements of m for which k>lambda
        m_max = m_max[:idxs_max, :]
        m_min = m_min[:idxs_min, :]

        # Find peaks
        # - column-wise summation
        m_max_sum = torch.sum(~m_max, dim=0)
        m_min_sum = torch.sum(~m_min, dim=0)
        p = torch.where(m_max_sum==0)[0] / srate
        t = torch.where(m_min_sum==0)[0] / srate

        return torch.diff(p), torch.diff(t), p

    
    
    def reshape_outputData_model(x_out):
        # unroll the windowed data
        nWindows, nSamples = np.shape(x_out)
        return np.reshape(x_out, newshape=(nWindows*nSamples, ))










    
                    