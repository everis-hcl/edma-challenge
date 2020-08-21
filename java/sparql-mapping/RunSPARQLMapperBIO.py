import subprocess
import os
import fnmatch
for subdir, dirs, files in os.walk("..\..\data\TFIDFcorpus\BIO"):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if fnmatch.fnmatch(filepath, '*WDcorpus*.txt'):
            p = subprocess.check_call(['java', '-jar', 'target\sparql-mapper-0.0.1-SNAPSHOT.jar', filepath, "..\..\data\ClassificationResults\BIO", 'BIO'], shell = True)
            if p == 0:
                print("Mapping is successful for input file:",filepath)