#!/bin/bash
echo "Starting the test"

resultsFile="test.txt"


echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_hpFiltered -f 0 -g 0 -m 4 -o sgd -e 10 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_hpFiltered -f 0 -g 0 -m 4 -o adam -e 10 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_hpFiltered -f 0 -g 0 -m 4 -o rmsprop -e 10 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_hpFiltered -f 0 -g 0 -m 4 -o adadelta -e 10 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_hpFiltered -f 0 -g 0 -m 4 -o sgd -e 25 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_hpFiltered -f 0 -g 0 -m 4 -o adam -e 25 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_hpFiltered -f 0 -g 0 -m 4 -o rmsprop -e 25 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_hpFiltered -f 0 -g 0 -m 4 -o adadelta -e 25 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_morphed -f 0 -g 0 -m 4 -o sgd -e 10 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_morphed -f 0 -g 0 -m 4 -o adam -e 10 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_morphed -f 0 -g 0 -m 4 -o rmsprop -e 10 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_morphed -f 0 -g 0 -m 4 -o sgd -e 25 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_morphed -f 0 -g 0 -m 4 -o adam -e 25 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_morphed -f 0 -g 0 -m 4 -o rmsprop -e 25 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
