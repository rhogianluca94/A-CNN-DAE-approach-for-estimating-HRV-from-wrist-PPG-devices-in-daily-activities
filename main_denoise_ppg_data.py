import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PPGdenoising import PPGdenoising



#---------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # load sample data
    PPG_PATH = Path(Path.cwd() / 'proposed' / 'sampleData.csv')
    csvData_ppg = pd.read_csv(PPG_PATH, sep=',', low_memory=False)
    
    # Prepare sample data
    ppgdenoising = PPGdenoising(srateOrig=64)     
    ppg_raw = csvData_ppg['ppgData']
    ppg_prep = ppgdenoising.prepData(data_ppg=ppg_raw, winLen=8, overlap=0)            
    
    # denoise data
    ppg_den, ppg_noisy = ppgdenoising.runDenoisingModel(ppg_data=ppg_prep)
    
    plt.figure(1)
    plt.plot(ppg_den, label='PPG Denoised', color='red')
    plt.plot(ppg_noisy, label='PPG Raw', color='blue')

    plt.title('PPG Signals: Denoised vs Noisy')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # apply tracking algorithm to PRV
    hr, stdrr, rmssd = ppgdenoising.getPRVfeatures(ppg_data=ppg_den, step=None)
    
    fig, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(hr, color='green')
    axs[0].set_title('meanHR')
    axs[0].grid(True)

    axs[1].plot(stdrr, color='orange')
    axs[1].set_title('stdRR')
    axs[1].grid(True)

    axs[2].plot(rmssd, color='purple')
    axs[2].set_title('RMSSD')
    axs[2].set_xlabel('samples (8sec-averages)') 
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()