cd train
#./script.sh
cd ../test
./script.sh
cd ..
python data_consolidation.py
python solution.py