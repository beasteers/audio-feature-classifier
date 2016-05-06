#!/usr/bin/env python
import essentia.standard as essentia
from afe import *
import glob


action = safe_get(sys.argv, 1)
model_name = safe_get(sys.argv, 2)
options = sys.argv[3:]


if not action:
    raise SystemExit('Please Specify an action')
else: action = action.lower()

if action in ['gather', 'model', 'info', 'delete'] and not model_name:
    raise SystemExit('Please Specify a model')

############################
# Define Directory Patterns
############################
modelDirectory = 'models/'
directory = lambda name: modelDirectory + '{}/'.format(name)
statesFile = lambda name: directory(name)+'{}.states.pickle'.format(name)
vectorFile = lambda name: directory(name)+'{}.vector.pickle'.format(name)
hmmFile = lambda name, suffix='': directory(name)+'{}{}.hmm.json'.format(name, suffix)
hmmBackup =  lambda name, suffix='': directory(name)+'{}{}.hmm.backup'.format(name, suffix)


# Generate Essentia Algorithm Functions
Hanning = essentia.Windowing(type='hann')
Spectrum = essentia.Spectrum()
MFCC = essentia.MFCC()
Yin = essentia.PitchYinFFT()
Loudness = essentia.Loudness()


############################
# Features and Processors
############################
# Create Features
features = FeatureVector([
    Feature('f0', smooth=(.1**3, .2)),
    Feature('mfcc', length=13, yRange=(-1000,1000)),
    Feature('loudness')
], name=model_name, file=vectorFile(model_name))


#####################
# Feature Extraction
#####################
# Data stream -> features
# Feature vector generates vector
@FeatureExtractor(features.input)
def process(frame, *args):

    frame_spectrum = Spectrum(Hanning(frame))
    mfcc_bands, mfcc_coeffs = MFCC(frame_spectrum)

    f0, f0_confidence = Yin(frame_spectrum)
    loudness = Loudness(frame)


    return [
        features['f0'].smooth(f0, confidence=f0_confidence),
        mfcc_coeffs,
        loudness
    ]



if action == 'gather':
    # Create the directory to store the vector file if it doesn't exist
    parent_dir = directory(model_name)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # From afe.py - Runs processor on input stream
    listen(
        features=features,
        processor=process,
        save=True,
        # graphing=graphing
    )

elif action == 'model':
    saveDir = hmmFile(model_name)

    # Backup the previous
    if '-ow' not in options and os.path.isfile(saveDir):
        hmm_json = read(saveDir)
        backupDir = hmmBackup(model_name, '-{}'.format(int(time.time())) )
        save(hmm_json, backupDir)
        print 'Backed up old model to: ', backupDir

    files = glob.glob(in_cwd(vectorFile(model_name)))
    if len(files) == 0: raise SystemExit('Vector file not found.')

    print 'Building {} model...'.format(model_name)
    hmm = vector_to_hmm(model_name, files[0])

    save(hmm.to_json(), saveDir)
    print 'Saved to {}.'.format(saveDir)


elif action == 'classify':
    ##################
    # HMM Creation
    ##################
    # Empty container to store hmms
    hmms = {}
    print 'Loading model files...'
    files = glob.glob(cwd + modelDirectory + '*/*.hmm.json')
    if len(files) == 0: raise SystemExit('No models found.')

    # Create HMMs from json file
    for f in files:
        hmm = HiddenMarkovModel.from_json(read(f))
        hmms[hmm.name] = hmm

    print '{} HMMs ({}) loaded and ready.'.format(len(hmms), [name for name in hmms])

    ##################
    # Classifier
    ##################

    # classifier used to get most likely model from a list of models
    # lookback is the length of the previous observation stream to look at
    classifier = Classifier(hmms,
                            threshold=75,
                            lookback=10,
                            smoothing=(.1**3, .2)
                        )


    def classify(vector):
        states = classifier.evaluate(vector)
        same_line( 'states: '+', '.join(map(str, states)) )

    # Tells the feature vector object to call classify when it gets a new vector in
    features.register(classify)

    listen(
        features=features,
        processor=process,
        # graphing=graphing
    )

elif action == 'info':
    hmm = HiddenMarkovModel.from_json(read(hmmFile(model_name)))
    print
    print hmm.name
    print 'Transition Matrix:'
    print hmm.dense_transition_matrix()
    print

elif action == 'delete':
    delete = raw_input("delete model '{}'? y/[n]".format(model_name))
    if delete in ['y', '', 'yes', 'yep', 'of course', 'yes please', "I've never been so sure in my damn life."]:
        os.remove(in_cwd(hmmFile(model_name)))

elif action == 'clean':
    delete = raw_input("delete model collection '{}'? y/[n]".format(model_name))
    if delete in ['y', '', 'yes', 'yep', 'of course', 'yes please', "I've never been so sure in my damn life."]:
        os.remove(in_cwd(hmmFile(model_name)))