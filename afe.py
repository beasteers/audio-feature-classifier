import time, pyaudio, collections
import pyqtgraph as pg
from helpers import *


class KalmanFilter:
    def __init__(self, process_variance, estimated_measurement_variance):
        # Process Variance = amount that the feature changes actually
        self.process_variance = process_variance
        # Measurement Variance = amount that the sensor data changes
        self.estimated_measurement_variance = estimated_measurement_variance
        self.posteri_estimate = 0.
        self.posteri_error_estimate = 1.

    # can take into account the confidence of a new measurement - i.e. f0_confidence
    def update(self, measurement, confidence=None):
        if confidence is None: confidence = 0
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        blending_factor = priori_error_estimate / (priori_error_estimate + self.estimated_measurement_variance * (1-confidence))
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

    def get_latest(self):
        return self.posteri_estimate

    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)
        return self.get_latest()

# Class for single feature
# x property is for graphing cache only
class Feature:
    def __init__(self, name, length=1, smooth=None, verbose=False, xRange=None, yRange=None):
        if smooth: self.add_filter(*smooth)
        else: self.smooth = lambda s, *args, **kwargs: s

        self.name = name
        self._x = []
        self._y = []
        self.count = 0

        self.xRange = xRange
        self.yRange = yRange

        self.roll = False
        self.length = length
        self.clear = length != 1
        self.verbose=verbose

    def add_filter(self, *args, **kwargs):
        self.smooth = KalmanFilter(*args, **kwargs)
        return self

    @property
    def last(self):
        return self._y[-1]
    @property
    def all(self):
        return self._y

    def __len__(self):
        return self.length

    @property
    def x(self):
        x = self._x
        y = self.y
        if len(y) and not len(x):
            x = self._x = range(0, len(y))
        if len(x) < len(y):
            x.extend(range(x[-1]+1, x[-1] + len(y) - len(x) + 1))
        elif len(x) > len(y):
            x = self._x = x[len(x) - len(y):]
        return x
    @x.setter
    def x(self, x): self._x = x

    def update_x(self, x):
        if x is not None:
            if self.roll and not self.clear: self._x = roll(self._x, x)
            else: self._x.append(x)

    @property
    def y(self):
        if self.clear: return self._y[-1]
        else: return self._y
    @y.setter
    def y(self, y): self._y = y

    def xy(self): return self.x, self.y

    def input(self, data):
        if data is not None:
            # try input is a list
            try: y = data[:]
            # data is float value
            except TypeError: y = data
            if y is not None and y is not False:
                self._y.append(y)
        return self.last


# Graphs features using pyqtgraph
class FeatureVisualizer:
    def __init__(self, feats, cols=2):
        self.app = pg.Qt.QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title="Audio Features")
        self.win.resize(1000, 600)
        self.num_of_columns = cols

        self.roll = False
        self.tstart = time.time()

        self.features = feats
        self.last_n = feats.n

        self.curve = []
        for i, feat in enumerate(feats):
            plt = self.win.addPlot(title=feat.name)
            plt.showGrid(x=True, y=True)
            if feat.xRange or feat.yRange: plt.setRange(xRange=feat.xRange, yRange=feat.yRange)
            self.curve.append(plt.plot(feat._x, feat._y, pen=(255, 0, 0)))
            if (i+1)%self.num_of_columns == 0: self.win.nextRow()

    @property
    def fresh(self, *args):
        return self.last_n < self.features.n

    def make_stale(self):
        self.last_n = self.features.n

    def graph(self, feature=None, i=None):
        if self.fresh:
            if feature or i:
                self._graph(feature, i)
            else:
                for i in range(len(self.features)):
                    self._graph(i=i)

    def _graph(self, feature=None, i=None):
        if self.fresh:
            if i is None:
                i = self.feature(feature)
                if i is False: return False
            x,y = self.features[i].xy()
            if y is not False and len(y):
                self.curve[i].setData(x, y)
                return True
        return False

    def feature(self, name):
        for i, f in enumerate(self.features):
            if f.name is name: return i
        return False

    def process_events(self):
        if self.fresh:
            self.make_stale()
            self.app.processEvents()

    def exit(self):
        self.win.close()
        # self._exit = True

# Rolls a numpy array - acts as array with fixed length
def roll(a, b=None, count=None):
    inc = a[-1] - a[-2]
    try:
        if not count: count = len(b)
        if b is None: b = np.arange(a[-1], (a[-1] + count) * inc, inc)
    except TypeError:
        count = 1
        if b is None: b = list(a[-1] + inc)
    a = np.roll(a, -1 * count)
    a[len(a) - count:] = b
    return a

# Decorator function to wrap feature extractors to convert pyaudio and send to inputs
def FeatureExtractor(*inputs):
    def inputDeligation(extractor):
        def dataPrep(data_string, frame_count, time_info, status):
            try:
                data = np.fromstring(data_string, dtype="Float32")
                data = np.nan_to_num(data)

                vector = extractor(data, frame_count, time_info, status)
                for i in inputs: i(vector)

                data_string = np.array(data).tostring()
                return (data_string, pyaudio.paContinue)
            except (KeyboardInterrupt, SystemExit):
                return (data_string, pyaudio.paComplete)
        return dataPrep
    return inputDeligation


class FeatureVector:
    def __init__(self, features, name=None, vectors=None, file=None):
        self.features = features
        self.name = name
        if vectors is None: self.vectors = []
        else: self.vectors = vectors

        # if not file and name: file = vectorFile(name)
        self.file = file
        self.load_vectors(file)

        self.l = sum([len(f) for f in features])
        self.outputs = []

    # Get features by index or feature name
    def __getitem__(self, item):
        if isinstance(item, basestring):
            for f in self.features:
                if f.name == item:
                    return f
        else:
            return self.features[item]

    def __iter__(self):
        for f in self.features: yield f

    # Number of features
    def __len__(self):
        return len(self.features)

    # All Feature Names
    @property
    def names(self):
        return [f.name for f in self.features]

    # Last vector
    @property
    def last(self):
        return self.vectors[-1]

    # Number of vectors collected
    @property
    def n(self):
        return len(self.vectors)

    def __str__(self):
        features = ', '.join(self.names)
        return 'Feature Vector ({} per vector, {} vectors): {}'.format(self.l, self.n, features)

    # Load previously gathered vectors
    def load_vectors(self, filename):
        if os.path.isfile(filename):
            data = read_pickle(filename)
            if data:
                self.vectors.extend(data['vectors'])
        self.initCount = self.n

    # Register output functions to send the vectors to when they come in
    def register(self, output):
        self.outputs.append(output)

    # Input the vector grouped by feature
    def input(self, vector):
        merged = []
        for i, feature in enumerate(self.features):
            val = feature.input(vector[i])
            try: merged.extend(val)
            except TypeError: merged.append(val)
        if len(merged) != self.l: raise ValueError('Vector must have {} values, but has {} values instead.'.format(self.l, len(merged)))
        self.vectors.append(merged)
        for out in self.outputs: out(merged)

    # Save feature vectors to file
    def save(self):
        if self.n > 0:
            if self.name:
                finalCount = self.n
                save = raw_input('Save {} features? ({} new) ([y]/n)?  '.format(finalCount, finalCount - self.initCount))
                if save.lower() in ['y', '', 'yes', 'yep', 'of course', 'yes please', 'yes, please!', 'eh why not', 'why not']:
                    data = {
                        'name': self.name,
                        'vectors': self.vectors
                    }
                    save_pickle(data, self.file)
                    print 'Saved {} new features.'.format(finalCount - self.initCount)
                else: print 'Features not saved.'



class Classifier:
    def __init__(self, hmms, threshold, lookback=20, smoothing=None ):
        self.hmms = hmms
        self.threshold = threshold
        self.seq = collections.deque(maxlen=lookback)
        self.states = []

        self.filters = {}
        if smoothing:
            for name in hmms: self.filters[name] = KalmanFilter(*smoothing)
        else:
            for name in hmms: self.filters[name] = lambda x, *a, **k: x

    def evaluate(self, vector):
        probs = []
        self.seq.append(vector)
        for name, hmm in self.hmms.iteritems():
            prob = self.filters[name]( hmm.log_probability(self.seq) )
            probs.append( (name, prob) )

        probs.sort(key=lambda x: x[1], reverse=True)

        states = []
        for name, prob in probs:
            dif = probs[0][1] - prob
            if dif < self.threshold:
                states.append( (name, prob) )
            else: break
        return states




def listen(features, processor, graphing=True, Fs=44100, chunk=2**10, save=False):

    # Audio Initialization
    pa = pyaudio.PyAudio()
    sample_period = chunk/float(Fs)

    print
    print 'Window Size', chunk
    print 'Window Period', sample_period
    print 'Sample Rate', Fs
    print

    # if graphing is turned on, create graphing instance
    if graphing: afe = FeatureVisualizer(features, cols=1)

    ###########################
    # Opening Audio Stream
    ###########################

    if save: raw_input('Hit Enter to start.')

    stream = pa.open(format=pyaudio.paFloat32,
                     channels=1,
                     rate=Fs,
                     input=True,
                     # output=True,
                     frames_per_buffer=chunk * 2,
                     stream_callback=processor)

    print 'Listening...'
    stream.start_stream()


    ############################################
    # Keeping Python Busy, Possibly graphing idk
    ############################################
    try:
        while stream.is_active():
            try:
                if graphing:
                    afe.graph()
                    afe.process_events()
                else:
                    time.sleep(0.1)
            except (KeyboardInterrupt, SystemExit):
                print 'Closing...'
                break
    except (KeyboardInterrupt, SystemExit): pass


    ################
    # Cleaning Up
    ################
    # if graphing: afe.exit()
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print 'Closed.'

    ################################
    # Saving Vectors to Pickle File
    ################################
    if save: features.save()

    print("Everything Done!")

