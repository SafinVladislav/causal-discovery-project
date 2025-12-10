![Causal Discovery Orientation](orientation_video.mp4)

# Causal Discovery with Quasi-Interventions

Python project for learning fully oriented causal DAGs on discrete data by combining the PC algorithm (skeleton) with simulated quasi-interventions that orient remaining undirected edges.

For detailed explanation of the method, statistical tests, intervention strategies and examples, see `Documentation.pdf`.

## Installation & Running

### Google Colab
```bash
!pip install pgmpy networkx pandas numpy scipy matplotlib tqdm joblib bnlearn requests statsmodels lightgbm scikit-learn graphviz pydot

!python "/content/drive/MyDrive/causal-discovery-project/run_simulation.py"
```

### Windows
Double-click `setup_windows.bat`  
or run in terminal:
```bat
setup_windows.bat
```

Then:
```cmd
venv\Scripts\activate
python run_simulation.py
deactivate
```

### Linux / macOS
```bash
bash setup_linux.sh
```

Then:
```bash
source venv/bin/activate
python run_simulation.py
deactivate
```

VS Code: open folder → Ctrl+Shift+P → select the `venv` interpreter → run `run_simulation.py`.

## Running tests (development)
After activating the virtual environment:
```bash
python -m unittest tests.test_graph_utils
python -m unittest tests.test_intervention
python -m unittest tests.test_orienting_alg
python -m unittest tests.test_statistical_tests
```

See `Documentation.pdf` for full method description.

**Author**: Vladislav Safin 
**License**: MIT (feel free to change)  
**Year**: 2025
