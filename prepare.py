import numpy as np
import random
import pickle
import wfdb as wf
import pywt
from model import Record, WindowPair
from utils import find_consec

def get_window_pairs(record, feature_window_nbeats, leadtime_nbeats, forecast_window_nbeats, exclude_only = 'N'):

    wps = []   

    ai = record.get_beat_indices()                                  # all beats
    bi = record.get_beat_indices(exclude_only = exclude_only)       # bad beats

    an = np.arange(len(ai))
    bn = an[np.in1d(ai,bi)]
    
    offL = feature_window_nbeats + leadtime_nbeats + forecast_window_nbeats - 1
    offR = len(ai) - forecast_window_nbeats + 1
    bn = bn[ (bn >= offL) & (bn < offR) ]    

    avail = an[offL:offR]
    used  = []
    
    for b in bn:
        for pos in np.arange(0, forecast_window_nbeats, 1):
            w1L, w1R = b - pos, b + forecast_window_nbeats - pos - 1
            w2L, w2R = w1L - leadtime_nbeats - feature_window_nbeats, w1L - leadtime_nbeats - 1
            used.extend([w1L,w1R])
            s1 = record.signal['ch1']['values'][ai[w2L]:ai[w2R]]
            s2 = record.signal['ch2']['values'][ai[w2L]:ai[w2R]]
            #s1 = record.signal['ch1']['values'][ai[w1L]:ai[w1R]]     # (for testing only)
            #s2 = record.signal['ch2']['values'][ai[w1L]:ai[w1R]]     # (for testing only)
            signal = np.array([s1,s2]).T
            wp = WindowPair('arrhythmic', signal)
            wps.append(wp)
    
    avail = list(avail[~np.in1d(avail,used)])
    pruned = find_consec(avail, forecast_window_nbeats)

    if len(wps) > len(pruned):
        wps = random.sample(wps, len(pruned))
    else:
        pruned = random.sample(pruned, len(wps))
    
    for p in pruned:
        w1L, w1R = avail[p[0]], avail[p[1]]
        w2L, w2R = w1L - leadtime_nbeats - feature_window_nbeats, w1L - leadtime_nbeats - 1
        s1 = record.signal['ch1']['values'][ai[w2L]:ai[w2R]]
        s2 = record.signal['ch2']['values'][ai[w2L]:ai[w2R]]
        #s1 = record.signal['ch1']['values'][ai[w1L]:ai[w1R]]     # (for testing only)
        #s2 = record.signal['ch2']['values'][ai[w1L]:ai[w1R]]     # (for testing only)
        signal = np.array([s1,s2]).T
        wp = WindowPair('normal', signal)
        wps.append(wp)
                
    return wps
    
def split_train_test(records, test_frac):
    test_records = random.sample(records, np.int(test_frac * len(records)))
    train_records = [record for record in records if record not in test_records]
    return train_records, test_records

def pass_window_checks(feat, lead, fore):
    condition  = (feat >= 1) & (lead >= 0) & (fore >= 1)
    condition &= (feat == int(feat)) & (lead == int(lead)) & (fore == int(fore))
    #condition &= (feat + fore + lead) <= 500
    return condition

def main(db_dir = 'mitdb/', test_frac = 0.2, feature_nbeats = 10, lead_nbeats = 5, forecast_nbeats = 5):
    assert pass_window_checks(feature_nbeats, lead_nbeats, forecast_nbeats), 'invalid window size'
    with open('clean_records.pkl', 'rb') as input:
        clean_records = pickle.load(input)
    records = []
    for record in clean_records:
        if len(record.get_beat_indices(exclude_only=['N','?'])) == 0: continue    # if no arrhythmic beats, skip
        record.wps = get_window_pairs(record, feature_nbeats, lead_nbeats, forecast_nbeats)
        records.append(record)
    train_records, test_records = split_train_test(records, test_frac)
    return train_records, test_records

if __name__ == "__main__":
    main()
