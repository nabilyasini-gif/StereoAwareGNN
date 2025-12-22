"""
StereoGNN-BBB: Blood-Brain Barrier Permeability Predictor
State-of-the-Art Model: AUC 0.9612 (External Validation on B3DB)

Author: Nabil Yasini-Ardekani
GitHub: https://github.com/abinittio

Streamlit Cloud Deployment Version - Self-Contained
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import json
import base64
import io
import os
import tempfile

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="StereoGNN-BBB | BBB Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    st.error("RDKit not available. Please install: pip install rdkit")

# PyTorch Geometric imports
try:
    from torch_geometric.nn import GATv2Conv, TransformerConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .prediction-negative {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        color: white;
    }
    .prediction-moderate {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .metric-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #667eea;
        margin: 0.3rem 0;
    }
    .info-box {
        background: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #0066cc;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MODEL ARCHITECTURE (Self-contained)
# ============================================================================
if TORCH_GEOMETRIC_AVAILABLE:
    class StereoAwareEncoder(nn.Module):
        """Stereo-aware molecular encoder using GATv2 + Transformer."""

        def __init__(self, node_features=21, hidden_dim=128, num_layers=4, heads=4, dropout=0.1):
            super().__init__()
            self.node_features = node_features
            self.hidden_dim = hidden_dim

            # Input projection
            self.input_proj = nn.Sequential(
                nn.Linear(node_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

            # GATv2 layers
            self.gat_layers = nn.ModuleList()
            self.gat_norms = nn.ModuleList()

            for i in range(num_layers):
                in_channels = hidden_dim
                out_channels = hidden_dim // heads
                self.gat_layers.append(
                    GATv2Conv(in_channels, out_channels, heads=heads, dropout=dropout, add_self_loops=True)
                )
                self.gat_norms.append(nn.LayerNorm(hidden_dim))

            # Transformer layer
            self.transformer = TransformerConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
            self.transformer_norm = nn.LayerNorm(hidden_dim)

            self.dropout = nn.Dropout(dropout)

        def forward(self, x, edge_index, batch):
            x = self.input_proj(x)

            for gat, norm in zip(self.gat_layers, self.gat_norms):
                residual = x
                x = gat(x, edge_index)
                x = norm(x + residual)
                x = self.dropout(x)

            residual = x
            x = self.transformer(x, edge_index)
            x = self.transformer_norm(x + residual)

            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)

            return torch.cat([x_mean, x_max], dim=1)


    class BBBClassifier(nn.Module):
        """BBB classifier with stereo encoder."""

        def __init__(self, encoder, hidden_dim=128):
            super().__init__()
            self.encoder = encoder
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 1)
            )

        def forward(self, x, edge_index, batch):
            graph_embed = self.encoder(x, edge_index, batch)
            return self.classifier(graph_embed)


# ============================================================================
# MOLECULAR FEATURIZATION
# ============================================================================
def get_atom_features(atom):
    """Generate 21-dimensional atom features including stereochemistry."""
    features = []

    # Atomic number (one-hot, common atoms)
    atom_types = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C, N, O, F, P, S, Cl, Br, I
    atom_num = atom.GetAtomicNum()
    features.extend([1 if atom_num == t else 0 for t in atom_types])

    # Degree (0-5)
    features.append(min(atom.GetDegree(), 5) / 5.0)

    # Formal charge
    features.append((atom.GetFormalCharge() + 2) / 4.0)

    # Hybridization
    hyb = atom.GetHybridization()
    hyb_types = [Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3]
    features.extend([1 if hyb == h else 0 for h in hyb_types])

    # Aromaticity
    features.append(1 if atom.GetIsAromatic() else 0)

    # In ring
    features.append(1 if atom.IsInRing() else 0)

    # Stereochemistry features (6 features)
    chiral_tag = atom.GetChiralTag()
    features.append(1 if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED else 0)
    features.append(1 if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW else 0)
    features.append(1 if chiral_tag == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW else 0)

    # E/Z stereo (from bonds)
    has_ez = False
    is_e = False
    is_z = False
    for bond in atom.GetBonds():
        stereo = bond.GetStereo()
        if stereo in [Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOZ]:
            has_ez = True
            if stereo == Chem.rdchem.BondStereo.STEREOE:
                is_e = True
            else:
                is_z = True
    features.extend([1 if has_ez else 0, 1 if is_e else 0, 1 if is_z else 0])

    return features


def smiles_to_graph(smiles):
    """Convert SMILES to PyG Data object with 21-dim features."""
    if not RDKIT_AVAILABLE or not TORCH_GEOMETRIC_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))

    x = torch.tensor(atom_features, dtype=torch.float)

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.extend([[i, j], [j, i]])

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


# ============================================================================
# DESCRIPTOR-BASED PREDICTOR (Fallback when no model weights)
# ============================================================================
class DescriptorBBBPredictor:
    """
    Descriptor-based BBB predictor using optimized rules.
    Based on published BBB penetration rules and trained coefficients.
    """

    def __init__(self):
        # Optimized coefficients from training on BBBP dataset
        self.coefficients = {
            'intercept': 0.65,
            'mw': -0.0012,      # Negative: higher MW = less penetration
            'logp': 0.08,       # Positive: higher logP = more penetration
            'tpsa': -0.008,     # Negative: higher TPSA = less penetration
            'hbd': -0.12,       # Negative: more H-donors = less penetration
            'hba': -0.05,       # Negative: more H-acceptors = less penetration
            'rotatable': -0.02, # Negative: more flexibility = less penetration
            'aromatic_rings': 0.05,
            'n_atoms': -0.005,
        }

    def predict(self, smiles):
        """Predict BBB permeability from SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES"

        # Calculate descriptors
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rotatable = Descriptors.NumRotatableBonds(mol)
        aromatic_rings = Descriptors.NumAromaticRings(mol)
        n_atoms = mol.GetNumAtoms()

        # Calculate score
        score = self.coefficients['intercept']
        score += self.coefficients['mw'] * (mw - 300) / 100
        score += self.coefficients['logp'] * (logp - 2)
        score += self.coefficients['tpsa'] * (tpsa - 60)
        score += self.coefficients['hbd'] * hbd
        score += self.coefficients['hba'] * (hba - 4)
        score += self.coefficients['rotatable'] * rotatable
        score += self.coefficients['aromatic_rings'] * aromatic_rings
        score += self.coefficients['n_atoms'] * (n_atoms - 25)

        # Sigmoid to get probability
        prob = 1 / (1 + np.exp(-score * 2))

        # Clamp to reasonable range
        prob = max(0.05, min(0.95, prob))

        return prob, None


# ============================================================================
# STEREOISOMER ENUMERATION
# ============================================================================
def enumerate_stereoisomers(smiles, max_isomers=16):
    """Enumerate all stereoisomers for a molecule."""
    if not RDKIT_AVAILABLE:
        return [smiles]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]

    opts = StereoEnumerationOptions(
        tryEmbedding=True,
        unique=True,
        maxIsomers=max_isomers
    )

    try:
        isomers = list(EnumerateStereoisomers(mol, options=opts))
        if len(isomers) == 0:
            return [smiles]
        return [Chem.MolToSmiles(iso, isomericSmiles=True) for iso in isomers]
    except:
        return [smiles]


# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_model():
    """Load the BBB model or fallback to descriptor predictor."""

    # First try to load GNN model with weights
    if TORCH_GEOMETRIC_AVAILABLE:
        try:
            encoder = StereoAwareEncoder(node_features=21, hidden_dim=128, num_layers=4)
            model = BBBClassifier(encoder, hidden_dim=128)

            # Try to load weights from various locations
            possible_dirs = [
                Path(__file__).parent / 'models',
                Path('.') / 'models',
                Path.home() / 'BBB_System' / 'models',
            ]

            model_files = [
                'bbb_stereo_v2_best.pth',
                'bbb_stereo_v2_fold4_best.pth',
                'bbb_stereo_v2_fold5_best.pth',
                'bbb_stereo_fold4_best.pth',
                'bbb_stereo_fold5_best.pth',
            ]

            for model_dir in possible_dirs:
                for mf in model_files:
                    model_path = model_dir / mf
                    if model_path.exists():
                        try:
                            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
                            model.load_state_dict(state_dict)
                            model.eval()
                            return {'type': 'gnn', 'model': model, 'name': mf}, None
                        except Exception as e:
                            continue
        except Exception as e:
            pass

    # Fallback to descriptor-based predictor
    if RDKIT_AVAILABLE:
        predictor = DescriptorBBBPredictor()
        return {'type': 'descriptor', 'model': predictor, 'name': 'Descriptor-Based (Fallback)'}, None

    return None, "No prediction method available"


# ============================================================================
# PREDICTION
# ============================================================================
def predict_single(model_info, smiles):
    """Predict BBB permeability for a single SMILES."""

    if model_info['type'] == 'gnn':
        model = model_info['model']
        graph = smiles_to_graph(smiles)
        if graph is None:
            return None, "Invalid SMILES"

        if graph.x.shape[1] != 21:
            return None, f"Feature mismatch: expected 21, got {graph.x.shape[1]}"

        graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)

        with torch.no_grad():
            logit = model(graph.x, graph.edge_index, graph.batch)
            prob = torch.sigmoid(logit).item()

        return prob, None

    elif model_info['type'] == 'descriptor':
        return model_info['model'].predict(smiles)

    return None, "Unknown model type"


def predict_with_stereo_enumeration(model_info, smiles):
    """Predict with stereoisomer enumeration."""
    isomers = enumerate_stereoisomers(smiles)

    predictions = []
    for iso in isomers:
        prob, err = predict_single(model_info, iso)
        if prob is not None:
            predictions.append((iso, prob))

    if not predictions:
        return None, "All stereoisomers failed"

    probs = [p[1] for p in predictions]

    return {
        'mean': np.mean(probs),
        'min': np.min(probs),
        'max': np.max(probs),
        'std': np.std(probs) if len(probs) > 1 else 0,
        'n_isomers': len(predictions),
        'predictions': predictions
    }, None


# ============================================================================
# MOLECULAR PROPERTIES
# ============================================================================
def get_properties(smiles):
    """Calculate molecular properties."""
    if not RDKIT_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    props = {
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'rotatable': Descriptors.NumRotatableBonds(mol),
        'formula': rdMolDescriptors.CalcMolFormula(mol),
        'atoms': mol.GetNumAtoms(),
    }

    # BBB rules (based on literature)
    props['rules'] = {
        'mw': 150 <= props['mw'] <= 500,
        'logp': 0 <= props['logp'] <= 5,
        'tpsa': props['tpsa'] <= 90,
        'hbd': props['hbd'] <= 3,
        'hba': props['hba'] <= 7,
    }
    props['rules_passed'] = sum(props['rules'].values())

    return props


def mol_to_image(smiles, size=(350, 250)):
    """Generate molecule image."""
    if not RDKIT_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        img_data = drawer.GetDrawingText()
        b64 = base64.b64encode(img_data).decode()
        return f"data:image/png;base64,{b64}"
    except:
        return None


# ============================================================================
# COMMON MOLECULES DATABASE
# ============================================================================
MOLECULES = {
    "caffeine": ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),
    "aspirin": ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),
    "morphine": ("CN1CC[C@]23[C@H]4Oc5c(O)ccc(C[C@@H]1[C@@H]2C=C[C@@H]4O)c35", "Morphine"),
    "cocaine": ("COC(=O)[C@H]1[C@@H]2CC[C@H](C2)N1C", "Cocaine"),
    "dopamine": ("NCCc1ccc(O)c(O)c1", "Dopamine"),
    "serotonin": ("NCCc1c[nH]c2ccc(O)cc12", "Serotonin"),
    "ethanol": ("CCO", "Ethanol"),
    "glucose": ("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "Glucose"),
    "diazepam": ("CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13", "Diazepam"),
    "thc": ("CCCCCc1cc(O)c2[C@@H]3C=C(C)CC[C@H]3C(C)(C)Oc2c1", "THC"),
    "nicotine": ("CN1CCC[C@H]1c2cccnc2", "Nicotine"),
    "melatonin": ("CC(=O)NCCc1c[nH]c2ccc(OC)cc12", "Melatonin"),
    "ibuprofen": ("CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O", "Ibuprofen"),
    "acetaminophen": ("CC(=O)Nc1ccc(O)cc1", "Acetaminophen"),
    "fentanyl": ("CCC(=O)N(c1ccccc1)[C@@H]2CCN(CCc3ccccc3)CC2", "Fentanyl"),
    "heroin": ("CC(=O)O[C@H]1C=C[C@H]2[C@H]3CC4=C5C(=C(OC(C)=O)C=C4C[C@@H]1[C@]23C)OCO5", "Heroin"),
    "lsd": ("CCN(CC)C(=O)[C@H]1CN([C@@H]2Cc3cn(C)c4cccc(C2=C1)c34)C", "LSD"),
    "mdma": ("CC(NC)Cc1ccc2OCOc2c1", "MDMA"),
    "ketamine": ("CNC1(CCCCC1=O)c2ccccc2Cl", "Ketamine"),
    "psilocybin": ("CN(C)CCc1c[nH]c2cccc(OP(=O)(O)O)c12", "Psilocybin"),
    "atenolol": ("CC(C)NCC(O)COc1ccc(CC(N)=O)cc1", "Atenolol"),
    "metformin": ("CN(C)C(=N)NC(=N)N", "Metformin"),
    "penicillin": ("CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O", "Penicillin"),
    "amoxicillin": ("CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O", "Amoxicillin"),
}


def resolve_input(user_input):
    """Resolve user input to SMILES."""
    if not user_input:
        return None, None, "Please enter a molecule"

    if not RDKIT_AVAILABLE:
        return None, None, "RDKit not available"

    text = user_input.strip()

    # Check if valid SMILES
    if Chem.MolFromSmiles(text) is not None:
        return text, "Custom Molecule", None

    # Check database (case-insensitive)
    key = text.lower().strip()
    if key in MOLECULES:
        return MOLECULES[key][0], MOLECULES[key][1], None

    return None, None, f"Could not resolve '{text}'. Enter a valid SMILES or drug name."


# ============================================================================
# FILE UPLOAD & BATCH PROCESSING
# ============================================================================
def parse_uploaded_file(uploaded_file):
    """Parse uploaded file and extract SMILES strings."""
    filename = uploaded_file.name.lower()
    smiles_list = []
    
    try:
        if filename.endswith('.csv'):
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            # Look for SMILES column (case-insensitive)
            smiles_col = None
            for col in df.columns:
                if col.lower() in ['smiles', 'smile', 'smi', 'structure']:
                    smiles_col = col
                    break
            
            if smiles_col is None:
                # Try to use the first column
                smiles_col = df.columns[0]
            
            # Extract SMILES and optional names
            for idx, row in df.iterrows():
                smiles = str(row[smiles_col]).strip()
                if smiles and smiles.lower() != 'nan':
                    # Try to get name from second column if available
                    name = None
                    if len(df.columns) > 1:
                        name_col = [c for c in df.columns if c.lower() in ['name', 'compound', 'molecule', 'id']]
                        if name_col:
                            name = str(row[name_col[0]]).strip()
                        elif df.columns[1] != smiles_col:
                            name = str(row[df.columns[1]]).strip()
                    
                    if not name or name.lower() == 'nan':
                        name = f"Molecule_{idx+1}"
                    
                    smiles_list.append((smiles, name))
        
        elif filename.endswith('.txt'):
            # Read text file (one SMILES per line)
            content = uploaded_file.read().decode('utf-8')
            lines = content.strip().split('\n')
            for idx, line in enumerate(lines):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Check if line has tab or comma separator
                    if '\t' in line:
                        parts = line.split('\t')
                        smiles = parts[0].strip()
                        name = parts[1].strip() if len(parts) > 1 else f"Molecule_{idx+1}"
                    elif ',' in line:
                        parts = line.split(',')
                        smiles = parts[0].strip()
                        name = parts[1].strip() if len(parts) > 1 else f"Molecule_{idx+1}"
                    else:
                        smiles = line
                        name = f"Molecule_{idx+1}"
                    
                    if smiles:
                        smiles_list.append((smiles, name))
        
        elif filename.endswith('.sdf'):
            # Read SDF file using RDKit
            if RDKIT_AVAILABLE:
                # Write to temporary file for RDKit to read
                # SDF files can be text, but we'll handle both cases
                try:
                    content = uploaded_file.read()
                    # Try to decode as UTF-8, if it fails, keep as bytes
                    if isinstance(content, bytes):
                        try:
                            content_str = content.decode('utf-8')
                        except UnicodeDecodeError:
                            content_str = content.decode('latin-1')
                    else:
                        content_str = content
                    
                    # Use context manager for proper cleanup
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as tmp:
                        tmp.write(content_str)
                        tmp_path = tmp.name
                    
                    try:
                        suppl = Chem.SDMolSupplier(tmp_path)
                        
                        for idx, mol in enumerate(suppl):
                            if mol is not None:
                                smiles = Chem.MolToSmiles(mol)
                                # Try to get name from molecule properties
                                name = mol.GetProp('_Name') if mol.HasProp('_Name') else f"Molecule_{idx+1}"
                                if not name or name.strip() == '':
                                    name = f"Molecule_{idx+1}"
                                smiles_list.append((smiles, name))
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass  # Ignore errors during cleanup
                except Exception as e:
                    return None, f"Error processing SDF file: {str(e)}"
            else:
                return None, "RDKit required for SDF files"
        
        else:
            return None, f"Unsupported file format: {filename}"
        
        if not smiles_list:
            return None, "No valid SMILES found in file"
        
        return smiles_list, None
    
    except Exception as e:
        return None, f"Error parsing file: {str(e)}"


def batch_predict(model_info, smiles_list, enumerate_stereo=True, progress_bar=None):
    """Batch predict BBB permeability for multiple molecules."""
    results = []
    
    for idx, (smiles, name) in enumerate(smiles_list):
        if progress_bar:
            progress_bar.progress((idx + 1) / len(smiles_list))
        
        # Validate SMILES
        if not RDKIT_AVAILABLE or Chem.MolFromSmiles(smiles) is None:
            results.append({
                'name': name,
                'smiles': smiles,
                'score': None,
                'category': 'Error',
                'error': 'Invalid SMILES'
            })
            continue
        
        # Predict
        try:
            if enumerate_stereo:
                result, err = predict_with_stereo_enumeration(model_info, smiles)
                if result:
                    score = result['mean']
                else:
                    score = None
            else:
                score, err = predict_single(model_info, smiles)
            
            if score is not None:
                if score >= 0.6:
                    category = 'BBB+'
                elif score >= 0.4:
                    category = 'BBB+/-'
                else:
                    category = 'BBB-'
                
                results.append({
                    'name': name,
                    'smiles': smiles,
                    'score': score,
                    'category': category,
                    'error': None
                })
            else:
                results.append({
                    'name': name,
                    'smiles': smiles,
                    'score': None,
                    'category': 'Error',
                    'error': err or 'Prediction failed'
                })
        except Exception as e:
            results.append({
                'name': name,
                'smiles': smiles,
                'score': None,
                'category': 'Error',
                'error': str(e)
            })
    
    return results


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">StereoGNN-BBB</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Blood-Brain Barrier Permeability Predictor | State-of-the-Art Performance</p>', unsafe_allow_html=True)

    # Check dependencies
    if not RDKIT_AVAILABLE:
        st.error("RDKit is not installed. Please install it with: pip install rdkit")
        st.stop()

    # Load model
    model_info, error = load_model()

    if error:
        st.error(f"Model loading failed: {error}")
        st.stop()

    # Show model info
    is_gnn = model_info['type'] == 'gnn'

    # Sidebar
    with st.sidebar:
        st.header("Model Info")

        if is_gnn:
            st.success(f"GNN Model: {model_info['name']}")
            st.markdown("**Performance (External Validation):**")
            st.metric("AUC", "0.9612")
            st.metric("Sensitivity", "97.96%")
            st.metric("Specificity", "65.25%")
        else:
            st.warning(f"Mode: {model_info['name']}")
            st.markdown("""
            <div class="info-box">
            Using descriptor-based prediction.<br>
            For full GNN accuracy, upload model weights to models/ folder.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Interpretation")
        st.success("BBB+ (>=0.6): Crosses BBB")
        st.warning("Moderate (0.4-0.6)")
        st.error("BBB- (<0.4): Does not cross")

        st.markdown("---")
        st.subheader("Features")
        st.markdown("""
        - Stereo-aware predictions
        - Stereoisomer enumeration
        - Batch file upload (CSV/TXT/SDF)
        - Molecular property analysis
        - BBB rule assessment
        """)

        st.markdown("---")
        st.markdown("**Author:** Nabil Yasini-Ardekani")
        st.markdown("[GitHub](https://github.com/abinittio)")

    # Main input
    st.subheader("Enter Molecule")

    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "SMILES or drug name",
            placeholder="e.g., Caffeine, Aspirin, Morphine, or enter SMILES",
            label_visibility="collapsed"
        )
    with col2:
        predict_btn = st.button("Predict", type="primary", use_container_width=True)

    # Quick examples
    st.markdown("**Quick Examples:**")
    examples = ["Caffeine", "Morphine", "THC", "Dopamine", "Glucose", "Atenolol"]
    cols = st.columns(6)
    for i, ex in enumerate(examples):
        with cols[i]:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state['mol_input'] = ex
                st.rerun()

    if 'mol_input' in st.session_state:
        user_input = st.session_state['mol_input']
        del st.session_state['mol_input']
        predict_btn = True

    # Stereo enumeration option
    enumerate_stereo = st.checkbox("Enumerate stereoisomers", value=True,
                                   help="Predict all possible stereoisomers and show range")

    # File upload section
    st.markdown("---")
    st.subheader("Or Upload File for Batch Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload a file containing SMILES",
        type=['csv', 'txt', 'sdf'],
        help="Upload CSV (with SMILES column), TXT (one SMILES per line), or SDF file"
    )
    
    if uploaded_file is not None:
        st.info(f"📄 File: **{uploaded_file.name}** ({uploaded_file.size} bytes)")
        
        # Parse file
        with st.spinner("Parsing file..."):
            smiles_list, parse_err = parse_uploaded_file(uploaded_file)
        
        if parse_err:
            st.error(f"Error: {parse_err}")
            st.stop()
        
        st.success(f"✅ Found {len(smiles_list)} molecules")
        
        # Show preview
        with st.expander("Preview molecules (first 10)"):
            preview_data = []
            for smiles, name in smiles_list[:10]:
                preview_data.append({'Name': name, 'SMILES': smiles})
            st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
        
        # Batch predict button
        if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
            st.markdown("---")
            st.subheader("Batch Prediction Results")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run batch prediction
            with st.spinner(f"Predicting {len(smiles_list)} molecules..."):
                results = batch_predict(model_info, smiles_list, enumerate_stereo, progress_bar)
            
            progress_bar.empty()
            status_text.empty()
            
            # Summary statistics
            successful = [r for r in results if r['error'] is None]
            failed = [r for r in results if r['error'] is not None]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", len(results))
            with col2:
                st.metric("Successful", len(successful), delta=None)
            with col3:
                st.metric("Failed", len(failed), delta=None)
            with col4:
                if successful:
                    avg_score = np.mean([r['score'] for r in successful])
                    st.metric("Avg Score", f"{avg_score:.3f}")
            
            # Category distribution
            if successful:
                st.markdown("### Category Distribution")
                categories = {}
                for r in successful:
                    cat = r['category']
                    categories[cat] = categories.get(cat, 0) + 1
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("BBB+", categories.get('BBB+', 0))
                with col2:
                    st.metric("BBB+/-", categories.get('BBB+/-', 0))
                with col3:
                    st.metric("BBB-", categories.get('BBB-', 0))
            
            # Results table
            st.markdown("### Detailed Results")
            
            results_df = pd.DataFrame([
                {
                    'Name': r['name'],
                    'SMILES': r['smiles'],
                    'BBB Score': f"{r['score']:.4f}" if r['score'] is not None else 'N/A',
                    'Category': r['category'],
                    'Error': r['error'] if r['error'] else ''
                }
                for r in results
            ])
            
            st.dataframe(results_df, use_container_width=True, height=400)
            
            # Download batch results
            st.markdown("---")
            st.subheader("Download Batch Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV export
                csv_data = []
                for r in results:
                    csv_data.append({
                        'name': r['name'],
                        'smiles': r['smiles'],
                        'bbb_score': r['score'] if r['score'] is not None else '',
                        'category': r['category'],
                        'error': r['error'] if r['error'] else '',
                        'model_type': model_info['type'],
                        'model_name': model_info['name'],
                        'timestamp': datetime.now().isoformat()
                    })
                
                csv_df = pd.DataFrame(csv_data)
                st.download_button(
                    "📥 Download CSV",
                    csv_df.to_csv(index=False),
                    "batch_bbb_predictions.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # JSON export
                json_data = {
                    'summary': {
                        'total': len(results),
                        'successful': len(successful),
                        'failed': len(failed),
                        'model_type': model_info['type'],
                        'model_name': model_info['name'],
                        'timestamp': datetime.now().isoformat()
                    },
                    'results': [
                        {
                            'name': r['name'],
                            'smiles': r['smiles'],
                            'bbb_score': r['score'],
                            'category': r['category'],
                            'error': r['error']
                        }
                        for r in results
                    ]
                }
                
                st.download_button(
                    "📥 Download JSON",
                    json.dumps(json_data, indent=2),
                    "batch_bbb_predictions.json",
                    "application/json",
                    use_container_width=True
                )
            
            st.stop()
    
    st.markdown("---")

    if predict_btn and user_input:
        smiles, name, err = resolve_input(user_input)

        if err:
            st.error(err)
            st.stop()

        st.markdown(f"**{name}**: `{smiles}`")

        with st.spinner("Predicting..."):
            if enumerate_stereo:
                result, pred_err = predict_with_stereo_enumeration(model_info, smiles)
            else:
                prob, pred_err = predict_single(model_info, smiles)
                if prob is not None:
                    result = {'mean': prob, 'min': prob, 'max': prob, 'std': 0, 'n_isomers': 1}
                else:
                    result = None

            props = get_properties(smiles)
            img = mol_to_image(smiles)

        if pred_err:
            st.error(f"Prediction failed: {pred_err}")
            st.stop()

        st.markdown("---")

        # Results
        col1, col2, col3 = st.columns([1.2, 1, 1])

        score = result['mean']

        with col1:
            if score >= 0.6:
                card_class = "prediction-positive"
                category = "BBB+"
                interp = "HIGH permeability - likely crosses BBB"
            elif score >= 0.4:
                card_class = "prediction-moderate"
                category = "BBB+/-"
                interp = "MODERATE - may partially cross"
            else:
                card_class = "prediction-negative"
                category = "BBB-"
                interp = "LOW permeability - unlikely to cross"

            st.markdown(f"""
            <div class="prediction-card {card_class}">
                <h2 style="margin:0; font-size:2rem;">{category}</h2>
                <h1 style="margin:0.3rem 0; font-size:2.5rem;">{score:.4f}</h1>
                <p style="margin:0; font-size:0.9rem;">{interp}</p>
            </div>
            """, unsafe_allow_html=True)

            if result['n_isomers'] > 1:
                st.markdown(f"""
                <div class="metric-box">
                    <b>Stereoisomer Analysis ({result['n_isomers']} isomers)</b><br>
                    Range: {result['min']:.4f} - {result['max']:.4f}<br>
                    Std Dev: {result['std']:.4f}
                </div>
                """, unsafe_allow_html=True)

        with col2:
            if img:
                st.image(img, caption=name, use_container_width=True)
            else:
                st.info("Molecule image not available")

        with col3:
            if props:
                st.markdown(f"**Formula:** {props['formula']}")
                st.markdown(f"**MW:** {props['mw']:.1f} Da")
                st.markdown(f"**LogP:** {props['logp']:.2f}")
                st.markdown(f"**TPSA:** {props['tpsa']:.1f} A²")
                st.markdown(f"**H-Donors:** {props['hbd']}")
                st.markdown(f"**H-Acceptors:** {props['hba']}")

                rules_color = "green" if props['rules_passed'] >= 4 else "orange" if props['rules_passed'] >= 3 else "red"
                st.markdown(f"**BBB Rules:** :{rules_color}[{props['rules_passed']}/5 passed]")

        # Download section
        st.markdown("---")
        st.subheader("Export Results")

        report = {
            'molecule': name,
            'smiles': smiles,
            'bbb_score': round(score, 4),
            'category': category,
            'interpretation': interp,
            'n_stereoisomers': result['n_isomers'],
            'score_min': round(result['min'], 4),
            'score_max': round(result['max'], 4),
            'score_std': round(result['std'], 4),
            'model_type': model_info['type'],
            'model_name': model_info['name'],
            'timestamp': datetime.now().isoformat()
        }

        if props:
            report.update({
                'formula': props['formula'],
                'molecular_weight': round(props['mw'], 2),
                'logp': round(props['logp'], 2),
                'tpsa': round(props['tpsa'], 2),
                'h_donors': props['hbd'],
                'h_acceptors': props['hba'],
                'bbb_rules_passed': props['rules_passed'],
            })

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "Download JSON",
                json.dumps(report, indent=2),
                f"{name.replace(' ','_')}_bbb_prediction.json",
                "application/json",
                use_container_width=True
            )
        with col2:
            df = pd.DataFrame([report])
            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                f"{name.replace(' ','_')}_bbb_prediction.csv",
                "text/csv",
                use_container_width=True
            )
        with col3:
            # Create simple text report
            text_report = f"""BBB Permeability Prediction Report
=====================================
Molecule: {name}
SMILES: {smiles}
Score: {score:.4f}
Category: {category}
Interpretation: {interp}

Model: {model_info['name']}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Molecular Properties:
- Formula: {props['formula'] if props else 'N/A'}
- MW: {props['mw']:.1f if props else 'N/A'} Da
- LogP: {props['logp']:.2f if props else 'N/A'}
- TPSA: {props['tpsa']:.1f if props else 'N/A'} A²
- BBB Rules: {props['rules_passed'] if props else 'N/A'}/5 passed

Generated by StereoGNN-BBB
Author: Nabil Yasini-Ardekani
"""
            st.download_button(
                "Download TXT",
                text_report,
                f"{name.replace(' ','_')}_bbb_prediction.txt",
                "text/plain",
                use_container_width=True
            )

    # Footer with available molecules
    with st.expander("Available Drug Names (click to expand)"):
        drug_list = sorted(MOLECULES.keys())
        cols = st.columns(5)
        for i, drug in enumerate(drug_list):
            with cols[i % 5]:
                st.write(f"• {drug.capitalize()}")


if __name__ == "__main__":
    main()
