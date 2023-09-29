@echo off
REM --- Modify the script paths accordingly ---
call C:\Users\vladi\Miniconda3\condabin\activate.bat
call conda deactivate
call conda activate tracking
cd C:\Users\vladi\OneDrive\Documents\OocyteMaturityClassifier
python predict.py --input input.jpg --output result.txt
exit