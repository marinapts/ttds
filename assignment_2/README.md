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
wget https://www.inf.ed.ac.uk/teaching/courses/tts/CW/CW2/systems.zip
mkdir systems eval_results
unzip systems.zip -d systems
rm systems.zip
```
The contents of the systems.zip, which includes the system files S1.results to S6.results and qrels.txt are available under the *systems* directory.

### Run
The functionality is split into 2 files: **main.py** and the module **eval.py**.
Run ```python main.py``` to run the code which will create the evaluation files in the *eval_results* directory.
