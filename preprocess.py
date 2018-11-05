import numpy as np
from scipy.signal import medfilt, filtfilt, butter
import pickle
import sys
import os
import wfdb as wf
from model import Record

def remove_baseline(record):
    """ Remove baseline wander, calculated by applying two filters to signal:
        - median filter of width 200ms (200ms * 360Hz = 72) to remove QRS complexes, P waves
        - median filter of width 600ms (600ms * 360Hz = 216) to remove T-waves 
    """
    for ch in record.signal.keys():
        signal = record.signal[ch]['values']
        baseline = medfilt(signal, 72 + 1)
        baseline = medfilt(baseline, 216 + 1)
        record.signal[ch]['values'] = signal - baseline
    return record

def denoise(record):
    """ Denoise the signal using a Butterworth bandpass filter between 0.05Hz and 100Hz.
    """
    Fnyq = record.fs / 2.
    low = 0.05 / Fnyq
    hi = 100. / Fnyq
    b, a = butter(1, [low, hi], btype='band')
    for ch in record.signal.keys():
        signal = record.signal[ch]['values']
        denoised = filtfilt(b, a, signal)
        record.signal[ch]['values'] = denoised
    return record

def scale(record):
    """ Scale the signal.
    """
    for ch in record.signal.keys():
        signal = record.signal[ch]['values']
        scaled = (signal - np.mean(signal)) / np.std(signal)
        record.signal[ch]['values'] = scaled
    return record

def clean_record(record):
    #record = denoise(record)
    record = remove_baseline(record)
    record = scale(record)
    return record
    
def check_channels(record):
#    channels = [ch for ch in record.signal.keys()]
#    for ch in channels:
#        if record.signal[ch]['name'] not in ['MLII', 'V1']: return False
    if record.signal['ch1']['name'] != 'MLII': record.swap_channels()
    if record.signal['ch1']['name'] != 'MLII': return False
    return True

def dldb(db_dir):
    def check():
        choice = input('Directory not found. Download database into this directory? [Y/N] ').lower()
        yes = {'y','yes'}
        no  = {'n','no'}
        if choice in yes: return True
        elif choice in no: return False
        else: sys.stdout.write('Please respond with "Y" or "N".')
    def go(db_dir):
        print('Downloading database...')
        wf.dl_database('mitdb', os.path.join(os.getcwd(), db_dir))
        print('Download complete.')
    if check() == True: go(db_dir)

def main(db_dir = 'mitdb/'):
    if not os.path.exists(db_dir): dldb(db_dir)
    records = []
    for record_name in wf.get_record_list('mitdb/'):
        record = Record(db_dir, record_name)
        if check_channels(record) == False: continue
        record = clean_record(record)
        records.append(record)
    with open('clean_records.pkl', 'wb') as output:
        pickle.dump(records, output, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
