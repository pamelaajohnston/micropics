#!/bin/bash
echo "Starting the test"

resultsFile="test.txt"
GPU = 0


echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
CUDA_VISIBLE_DEVICES=${GPU} python3 processImages.py -d /home/1609098/micropics/data/workingSet_hpFiltered224/ -f 0 -g 0 -m 3 -o adam -e 7 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
CUDA_VISIBLE_DEVICES=0 python3 processImages.py -d /home/1609098/micropics/data/workingSet_hpFiltered224/ -f 0 -g 0 -m 3 -o sgd1 -e 7 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
CUDA_VISIBLE_DEVICES=0 python3 processImages.py -d /home/1609098/micropics/data/workingSet_hpFiltered224/ -f 0 -g 0 -m 3 -o rmsprop -e 7 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
CUDA_VISIBLE_DEVICES=0 python3 processImages.py -d /home/1609098/micropics/data/workingSet_hpFiltered224/ -f 0 -g 0 -m 3 -o adam -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
CUDA_VISIBLE_DEVICES=0 python3 processImages.py -d /home/1609098/micropics/data/workingSet_hpFiltered224/ -f 0 -g 0 -m 3 -o adam -e 40 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}


echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
CUDA_VISIBLE_DEVICES=0 python3 processImages.py -d /home/1609098/micropics/data/workingSet_hpFiltered224/ -f 0 -g 0 -m 3 -o sgd1 -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
CUDA_VISIBLE_DEVICES=0 python3 processImages.py -d /home/1609098/micropics/data/workingSet_hpFiltered224/ -f 0 -g 0 -m 3 -o sgd1 -e 40 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
CUDA_VISIBLE_DEVICES=0 python3 processImages.py -d /home/1609098/micropics/data/workingSet_hpFiltered224/ -f 0 -g 0 -m 3 -o rmsprop -e 20 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}

echo "Started at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
CUDA_VISIBLE_DEVICES=0 python3 processImages.py -d /home/1609098/micropics/data/workingSet_hpFiltered224/ -f 0 -g 0 -m 3 -o rmsprop -e 40 >> ${resultsFile}
echo "Finished/killed at:" >> ${resultsFile}
echo $(date) >> ${resultsFile}
