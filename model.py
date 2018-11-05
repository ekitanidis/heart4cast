import numpy as np
from scipy.signal import medfilt
import wfdb as wf
import pywt

class Record(object):
    
    def __init__(self, db_dr, record_name):
        signals, fields = wf.rdsamp(db_dr + record_name)
        annotations = wf.rdann(db_dr + record_name, 'atr', return_label_elements = ['symbol'])
        self.annotations = annotations
        self.name = record_name
        self.fs = fields['fs']
        self.length = fields['sig_len']
        self.time = np.arange(self.length) * 1. / self.fs
        self.signal = {'ch1' : {'values' : signals[:,0], 'name' : fields['sig_name'][0], 'units' : fields['units'][0]},
                       'ch2' : {'values' : signals[:,1], 'name' : fields['sig_name'][1], 'units' : fields['units'][1]}}

    def get_beat_indices(self, include_only = None, exclude_only = []):
        """ Returns indices in sample where there are beats annotations.
            Can specify which beats to include or exclude. Default is return all beats. 
        """  
        if include_only is None:
            include_only = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 
                            'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
        include_only = [beat for beat in include_only if beat not in exclude_only]
        beat_indices = self.annotations.sample[np.in1d(self.annotations.symbol, include_only)].astype(int)
        return beat_indices
        
    def swap_channels(self):
        """ Swap two channels. 
        """       
        ch1, ch2 = self.signal['ch1'], self.signal['ch2']
        self.signal['ch1'] = ch2
        self.signal['ch2'] = ch1
        self.annotations.chan = 1 - self.annotations.chan  # swap 0,1
        
    def flip_polarity(self, channel):
        """ Flip the polarity of a given channel. 
        """       
        channels = [k for k in self.signal.keys()]
        assert channel in channels, "channel not found. try: " + str(channels)
        flipped_values = self.signal[channel]['values']
        self.signal[channel]['values'] = -flipped_values

    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

    # TODO 
    #def check_unit_consistency(self):
    #    if self.signal['ch1']['units'] != self.signal['ch2']['units']:
    #        <do something>

class WindowPair:
    
    def __init__(self, label, signal):
        self.label = label
        self.signal = signal
        self.length = np.shape(signal)[0]
