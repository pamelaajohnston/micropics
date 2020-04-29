#!/bin/bash
echo "Starting the test"

resultsFile="test.txt"

workingDir = "/home/pam/data/micropics/workingSet_hpFiltered"
echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d ${workingDir} -f 0 -g 0 -m 4 -o sgd -e 30 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d ${workingDir} -f 0 -g 0 -m 4 -o adam -e 30 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d ${workingDir} -f 0 -g 0 -m 4 -o rmsprop -e 30 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d ${workingDir} -f 0 -g 0 -m 4 -o adadelta -e 30 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

workingDir = "/home/pam/data/micropics/workingSet_morphed"
echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d ${workingDir} -f 0 -g 0 -m 4 -o sgd -e 30 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d ${workingDir} -f 0 -g 0 -m 4 -o adam -e 30 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d ${workingDir} -f 0 -g 0 -m 4 -o rmsprop -e 30 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d ${workingDir} -f 0 -g 0 -m 4 -o adadelta -e 30 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

workingDir = "/home/pam/data/micropics/workingSet_hpFiltered"
echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d ${workingDir} -f 0 -g 0 -m 5 -o sgd -e 10 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d ${workingDir} -f 0 -g 0 -m 5 -o adam -e 10 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d ${workingDir} -f 0 -g 0 -m 5 -o rmsprop -e 10 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d ${workingDir} -f 0 -g 0 -m 5 -o adadelta -e 10 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
