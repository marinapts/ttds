# Text Technologies for Data Science
## Coursework 2 - IR Evaluation and Text Classification
https://www.inf.ed.ac.uk/teaching/courses/tts/CW/assignment2.html


### System Requirements
This project is written in Python and requires Python 3 to be installed.
My environment uses Python 3.7.3.
We create a [virtual environment](https://docs.python.org/3/tutorial/venv.html) so that the versions of the packages that are used are consistent across all machines.

Create the virtual environment using the *venv* command, which creates the hidden directory *.env*. Activate it and install the packages from requirements.txt.
```bash
python3 -m venv .env
source ./.env/bin/activate
pip install -r requirements.txt
```

### Data
Before running the code, download the data collection zip from the course's website, create a directory *data* and unzip the collection:
```bash
wget https://www.inf.ed.ac.uk/teaching/courses/tts/CW/CW1/CW1collection.zip
mkdir data
unzip CW1collection.zip -d data
rm CW1collection.zip
```
Now the data TREC xml and the queries are available in the data directory.

### Run
The functionality is split into 3 different files: main.py, preprocess.py and index_search.py. Run ```python main.py```
to run the code which will update the files in the *results* directory.



#### IR Evaluation

#### Text classification

#### Classification evaluation
The module evaluation.py expects 3 command line arguments when calling it: test_file, out_file and eval_file.
The files Eval.txt (or Eval2.txt) can be found in the outputs directory.
