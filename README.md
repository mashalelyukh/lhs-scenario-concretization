# Scenario Concretization Prototype

This prototype supports the **generation of concrete driving scenarios** from logical OpenSCENARIO 2.1.0 DSL files. It was developed as part of a bachelor thesis focusing on interpretable, modular scenario refinement for scenario-based testing of automated driving systems (ADS).

## Purpose of the Prototype

The goal of the system is to provide a **user-guided pipeline** for transitioning from logical scenario specifications to concrete test cases. It supports:

- Parameter sampling with **Latin Hypercube Sampling (LHS)**
- Iterative scenario refinement via **Bayesian Optimization**
- Expert input through **criticality labeling**
- Visualization and result inspection

The tool bridges the gap between structured formal definitions and simulator-ready scenarios.

---

## Features

- **Parsing** OpenSCENARIO DSL 2.1.0 logical files
- **Sampling** from float/int/enum parameter ranges
- **Comment handling, type casting, range flattening**
- **Scenario concretization** using templates
- **Mock criticality function** integration
- **Bayesian Optimization** for iterative refinement
- **Visualizations** for sampling and optimization behavior

---

## How to Run

```bash
python main.py
```

The CLI will guide you through:

1. Selecting an input logical scenario file  
2. Viewing extracted parameters  
3. Defining how many samples to generate  
4. Labeling the generated scenarios with a criticality score (0–1)  
5. Refining with Bayesian Optimization (BO)

---

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | CLI-based pipeline driver |
| `lark_parser.py` | DSL parsing logic |
| `lhs_sampler.py` | Latin Hypercube Sampling |
| `ranges_concretizer.py` | Scenario instantiation |
| `bayes_optimization.py` | Model-based refinement |
| `mock_functions.py` | Test functions |
| `visualization.py` | Plotting tools |
| `cli.py` | User prompts and display |
| `utils.py` | Helper utilities |
| `file_manager.py` | File system handling |

---

##  Dependencies

- Python ≥ 3.8  
- `scikit-learn`  
- `matplotlib`  
- `lark`  
- `numpy`

Install with:

```bash
pip install -r requirements.txt
```

##  Known Limitations

- BO visualization limited to 2D (1 continuous + 1 discrete param)  
- No automated testing; informal validation only  
- Abstract-level constraints (e.g., `vehicle A behind B`, or `speed>=20`) must be manually resolved



