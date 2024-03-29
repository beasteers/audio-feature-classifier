# Audio Feature Classifier
### CS480 Final Project

## Dependencies
Dependencies available through pip are listed in the requirements.txt file. To install run:
```
pip install -r requirements.txt
```
Essentia, the audio feature extraction library is unfortunately unavailable through pip, but can be installed through Homebrew using the tap below. For Linux and Windows, check the link for instructions on compiling from source.
#### Essentia on OS X (Homebrew)
Homebrew formulas for Essentia and Gaia installation
Install homebrew tap:
```
brew tap MTG/essentia
```
#### Other compiling options are described on [essentia’s site](http://essentia.upf.edu/documentation/installing.html).



### Usage
To start fresh, delete the models folder and start gathering data; otherwise try running classify. You could probably just delete everything in the models folder except for the background noise folder as the other models were trained on me. 

The script is meant to be used as a command line tool and all code blocks given are to be run in terminal.
## Gather data
Start by deciding on what you want to model (i.e. ’Normal’, ‘Distressed’).

For each model run: 
```
./project.py gather <name>
```
So for example: 
```
./project.py gather normal
```

Then teach the program what that state sounds like. When you’re finished, quit using Ctrl-C or whatever the KeyboardInterrupt for your system is. It may not always quit due to threading issues, but just keep trying and it’ll quit eventually. Then confirm whether or not you’re satisfied with the recording.

You can go back and record more by running the command with the same name and it will append the session to the previous one.

## Build models
Once the vectors have been collected, the models must be constructed. To do this, run: 
```
./project.py model <name>
```
To remodel it, run the same command. No need to delete the old file. The old file will be backed up by default.

## Classify
To receive any meaningful results, make sure you have at least two models as the classifier is comparative.

To run the classification type: 
```
./project.py classify
```
and it will run using all of the models found in the pattern ‘models/*/*.hmm.json’. To omit a model, just drag the parent folder to some other path that doesn’t match that pattern i.e. ‘models/storage/*/*.hmm.json’

####  Delete model
If you’re too lazy to find the *.hmm.json file in the directory (like me), you can just run 
```
./project.py delete <name>
```
and it will get rid of it for you. To rebuild the model just rerun the model command above.

####  Model info
If you want to look at a model transition matrix, just use: 
```
./project.py info <name>
```

The script can also be called as
```
/path/to/script/project.py …
or
python /path/to/script/project.py …
```