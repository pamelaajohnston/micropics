#!/bin/bash
echo "Starting the test"

resultsFile="test.txt"


echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -d /home/pam/data/micropics/workingSet_hpFiltered -f 0 -g 0 -m 1 -o sgd1 -e 10 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -d /home/pam/data/micropics/workingSet_hpFiltered -f 0 -g 0 -m 1 -o adam -e 10 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -d /home/pam/data/micropics/workingSet_hpFiltered -f 0 -g 0 -m 1 -o rmsprop -e 10 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -d /home/pam/data/micropics/workingSet_hpFiltered -f 0 -g 0 -m 1 -o sgd1 -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -d /home/pam/data/micropics/workingSet_hpFiltered -f 0 -g 0 -m 1 -o adam -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -d /home/pam/data/micropics/workingSet_hpFiltered -f 0 -g 0 -m 1 -o rmsprop -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -d /home/pam/data/micropics/workingSet_hpFiltered -f 0 -g 0 -m 1 -o sgd1 -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -d /home/pam/data/micropics/workingSet_hpFiltered -f 0 -g 0 -m 1 -o adam -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -d /home/pam/data/micropics/workingSet_hpFiltered224 -f 0 -g 0 -m 1 -o rmsprop -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -d /home/pam/data/micropics/workingSet_hpFiltered224 -f 0 -g 0 -m 2 -o sgd1 -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -d /home/pam/data/micropics/workingSet_hpFiltered224 -f 0 -g 0 -m 2 -o adam -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -d /home/pam/data/micropics/workingSet_hpFiltered224 -f 0 -g 0 -m 2 -o rmsprop -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -d /home/pam/data/micropics/workingSet_hpFiltered224 -f 0 -g 0 -m 2 -o sgd1 -e 10 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -d /home/pam/data/micropics/workingSet_hpFiltered224 -f 0 -g 0 -m 2 -o adam -e 10 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -d /home/pam/data/micropics/workingSet_hpFiltered224 -f 0 -g 0 -m 2 -o rmsprop -e 10 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
