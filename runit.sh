#!/bin/bash
echo "Starting the test"

resultsFile="test.txt"


echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_hpFiltered -f 0 -g 0 -m 1 -o sgd -e 3 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_resize/ -f 1 -g 0 -m 1 -o sgd -e 3 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_norm/ -f 2 -g 0 -m 1 -o sgd -e 3 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_hpFiltered224/ -f 3 -g 0 -m 2 -o sgd -e 3 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_resize224/ -f 1 -g 0 -m 2 -o sgd -e 3 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_norm224/ -f 2 -g 0 -m 2 -o sgd -e 3 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_hpFiltered224/ -f 0 -g 0 -m 3 -o sgd -e 3 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_resize224/ -f 0 -g 0 -m 3 -o sgd -e 3 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_norm224/ -f 0 -g 0 -m 3 -o sgd -e 3 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}



echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_hpFiltered -f 0 -g 0 -m 1 -o sgd -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_resize/ -f 1 -g 0 -m 1 -o sgd -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_norm/ -f 2 -g 0 -m 1 -o sgd -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_hpFiltered224/ -f 3 -g 0 -m 2 -o sgd -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_resize224/ -f 1 -g 0 -m 2 -o sgd -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_norm224/ -f 2 -g 0 -m 2 -o sgd -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_hpFiltered224/ -f 0 -g 0 -m 3 -o sgd -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_resize224/ -f 0 -g 0 -m 3 -o sgd -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
nohup python3 processImages.py -a /home/pam/data/micropics/after -b /home/pam/data/micropics/before -d /home/pam/data/micropics/workingSet_norm224/ -f 0 -g 0 -m 3 -o sgd -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
