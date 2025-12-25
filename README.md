# StereoGNN-BBB: Blood-Brain Barrier Permeability Predictor

A state-of-the-art Blood-Brain Barrier (BBB) permeability prediction tool powered by stereo-aware Graph Neural Networks (GNN).

## Features

- **Single Molecule Prediction**: Enter SMILES strings or drug names for instant predictions
- **Batch File Upload**: Upload CSV, TXT, or SDF files for batch predictions
- **Stereoisomer Enumeration**: Automatic enumeration and prediction of all stereoisomers
- **Molecular Property Analysis**: Comprehensive molecular descriptors and BBB rule assessment
- **Export Results**: Download predictions in JSON, CSV, or TXT format

## Performance

**External Validation on B3DB Dataset:**
- AUC: 0.9612
- Sensitivity: 97.96%
- Specificity: 65.25%

## Usage

### Single Molecule Prediction

1. Enter a SMILES string or drug name in the text input
2. Click "Predict" to get BBB permeability prediction
3. View detailed molecular properties and interpretation
4. Download results in your preferred format

### Batch File Upload

Upload files containing multiple molecules for batch predictions:

#### Supported File Formats

1. **CSV Files** (`.csv`)
   - Must have a column named "SMILES" (case-insensitive)
   - Optional second column for molecule names
   - Example:
     ```csv
     SMILES,Name
     CN1C=NC2=C1C(=O)N(C(=O)N2C)C,Caffeine
     CC(=O)Oc1ccccc1C(=O)O,Aspirin
     CCO,Ethanol
     ```

2. **Text Files** (`.txt`)
   - One SMILES per line
   - Tab or comma-separated SMILES and name
   - Example:
     ```
     CN1C=NC2=C1C(=O)N(C(=O)N2C)C	Caffeine
     CC(=O)Oc1ccccc1C(=O)O	Aspirin
     CCO	Ethanol
     ```

3. **SDF Files** (`.sdf`)
   - Standard Structure-Data File format
   - Molecule names extracted from the _Name property

#### Steps for Batch Prediction

1. Click "Upload a file containing SMILES"
2. Select your file (CSV, TXT, or SDF)
3. Review the preview of parsed molecules
4. Click "🚀 Run Batch Prediction"
5. View summary statistics and detailed results table
6. Download batch results in CSV or JSON format

## Interpretation

- **BBB+ (≥0.6)**: High permeability - likely crosses the blood-brain barrier
- **BBB+/- (0.4-0.6)**: Moderate permeability - may partially cross
- **BBB- (<0.4)**: Low permeability - unlikely to cross

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Required packages:
- streamlit>=1.28.0
- numpy>=1.24.0
- pandas>=2.0.0
- rdkit>=2023.9.1
- torch>=2.0.0
- torch-geometric>=2.4.0

### Running the Application

```bash
streamlit run app.py
```

## Model Information

The application supports two prediction modes:

1. **GNN Model** (when model weights are available):
   - Stereo-aware Graph Attention Networks (GATv2)
   - Transformer-based molecular encoding
   - State-of-the-art performance

2. **Descriptor-Based Model** (fallback):
   - Rule-based prediction using molecular descriptors
   - Optimized coefficients from BBBP dataset training

## Author

**Nabil Yasini-Ardekani**
- GitHub: [@abinittio](https://github.com/abinittio)

## License

This project is provided as-is for research and educational purposes.

## Citation

If you use this tool in your research, please cite:

```
StereoGNN-BBB: A Stereo-Aware Graph Neural Network for Blood-Brain Barrier Permeability Prediction
Nabil Yasini-Ardekani
```

## Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) for the web interface
- [RDKit](https://www.rdkit.org/) for cheminformatics
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) for graph neural networks
