import subprocess
import os
import fnmatch
for subdir, dirs, files in os.walk("..\..\data\TFIDFcorpus\GIT"):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if fnmatch.fnmatch(filepath, '*WDcorpus*.txt'):
            p = subprocess.check_call(['java', '-jar', 'target\sparql-mapper-0.0.1-SNAPSHOT.jar', filepath, "..\..\data\ClassificationResults\GIT", 'GIT'], shell = True)
            if p == 0:
                print("Mapping is successful for input file:",filepath)
