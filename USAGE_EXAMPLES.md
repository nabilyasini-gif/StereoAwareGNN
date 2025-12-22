# File Upload Feature - Usage Examples

This document provides examples of how to use the file upload feature for batch BBB predictions.

## Example 1: CSV File Format

Create a file named `molecules.csv`:

```csv
SMILES,Name
CN1C=NC2=C1C(=O)N(C(=O)N2C)C,Caffeine
CC(=O)Oc1ccccc1C(=O)O,Aspirin
CN1CCC[C@H]1c2cccnc2,Nicotine
NCCc1ccc(O)c(O)c1,Dopamine
CCO,Ethanol
OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O,Glucose
CC(C)NCC(O)COc1ccc(CC(N)=O)cc1,Atenolol
CN(C)C(=N)NC(=N)N,Metformin
```

### Steps:
1. Open the StereoGNN-BBB application
2. Scroll to "Or Upload File for Batch Prediction"
3. Click "Browse files" or drag and drop your `molecules.csv`
4. Review the preview showing the first 10 molecules
5. Click "🚀 Run Batch Prediction"
6. View results and download CSV or JSON report

## Example 2: Text File Format (Tab-separated)

Create a file named `molecules.txt`:

```
CN1C=NC2=C1C(=O)N(C(=O)N2C)C	Caffeine
CC(=O)Oc1ccccc1C(=O)O	Aspirin
CCO	Ethanol
NCCc1ccc(O)c(O)c1	Dopamine
```

### Steps:
Same as CSV - just upload the `.txt` file instead.

## Example 3: Text File Format (Simple, one SMILES per line)

Create a file named `simple_molecules.txt`:

```
CN1C=NC2=C1C(=O)N(C(=O)N2C)C
CC(=O)Oc1ccccc1C(=O)O
CCO
NCCc1ccc(O)c(O)c1
```

Molecules will be automatically named as "Molecule_1", "Molecule_2", etc.

## Example 4: SDF File Format

SDF (Structure-Data File) format is commonly used in chemistry. The application will:
- Read molecular structures from the SDF file
- Extract molecule names from the `_Name` property
- Convert structures to SMILES for prediction

## Understanding Results

After batch prediction, you'll see:

### Summary Statistics
- **Total**: Number of molecules processed
- **Successful**: Number of successful predictions
- **Failed**: Number of failed predictions (invalid SMILES)
- **Avg Score**: Average BBB permeability score

### Category Distribution
- **BBB+**: Molecules likely to cross the blood-brain barrier (score ≥ 0.6)
- **BBB+/-**: Molecules with moderate permeability (0.4 ≤ score < 0.6)
- **BBB-**: Molecules unlikely to cross (score < 0.4)

### Results Table
Detailed table showing:
- Molecule name
- SMILES string
- BBB Score (0.0000 to 1.0000)
- Category (BBB+, BBB+/-, BBB-)
- Error message (if prediction failed)

### Export Options
- **CSV**: Comma-separated values for spreadsheet applications
- **JSON**: Structured data format including summary statistics

## Tips

1. **Column Names**: For CSV files, the application looks for columns named:
   - SMILES: `smiles`, `smile`, `smi`, `structure` (case-insensitive)
   - Names: `name`, `compound`, `molecule`, `id` (case-insensitive)

2. **Large Files**: For very large files (>1000 molecules), the prediction may take several minutes. A progress bar will show the status.

3. **Error Handling**: Invalid SMILES will be marked as "Error" in the results but won't stop the processing of other molecules.

4. **Stereoisomers**: The "Enumerate stereoisomers" checkbox applies to batch predictions too. When enabled, the application will:
   - Generate all possible stereoisomers for each molecule
   - Predict BBB permeability for each stereoisomer
   - Report the average score

## Troubleshooting

### "No valid SMILES found in file"
- Check that your file contains valid SMILES strings
- Ensure the SMILES column is properly named (for CSV)
- Verify the file format matches the extension (.csv, .txt, .sdf)

### "Invalid SMILES" errors
- Some SMILES in your file may be malformed
- Check for special characters or truncated strings
- Validate SMILES using RDKit or an online tool

### "RDKit required for SDF files"
- SDF files require RDKit to be installed
- For Streamlit Cloud deployment, ensure rdkit is in requirements.txt

## Sample Test Files

You can create test files using the examples above. Here's a quick Python script to generate a test CSV:

```python
import pandas as pd

data = {
    'SMILES': [
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
        'CC(=O)Oc1ccccc1C(=O)O',         # Aspirin
        'CCO',                            # Ethanol
        'NCCc1ccc(O)c(O)c1',             # Dopamine
    ],
    'Name': ['Caffeine', 'Aspirin', 'Ethanol', 'Dopamine']
}

df = pd.DataFrame(data)
df.to_csv('test_molecules.csv', index=False)
print("Test file created: test_molecules.csv")
```

Save this as `create_test_file.py` and run: `python create_test_file.py`
