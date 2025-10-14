## Web Service

Try our web app for spectrum prediction and attention visualization without local setup:

**[🚀 Launch Web App](https://huggingface.co/spaces/liuzhijin/hdse-ms-attn-viz)**

### Features:
- Input SMILES and get predicted spectra online
- Visualize molecular attention weights (atom/bond-level heatmaps)  
- Choose among different pretrained models for quick comparison

# HDSE-MS Data Preparation and Usage

## Conda Installation

Open a terminal or command prompt and navigate to your project directory. Create a new conda environment named `HDSE-MS` and add the required dependencies:

```bash
conda create -n HDSE-MS python=3.9.20
```

Activate the newly created conda environment and install the necessary dependencies:

```bash
conda activate HDSE-MS
pip install -r requirements.txt
```

---

## Data Download

### Datasets

You can download public datasets from the following links:

- [Public Data (HuggingFace)](https://huggingface.co/datasets/liuzhijin/Tandem_mass_spectrum)  
  A collection of tandem mass spectra for machine learning and analysis.

- [MoNA](https://mona.fiehnlab.ucdavis.edu/)  
  A free, public repository of mass spectra from North American and international contributors.

- [MassBank](https://msbi.ipb-halle.de/MassBank/)  
  The first public repository of mass spectra for sharing and disseminating reference mass spectral data.

Extract the downloaded ZIP file(s) into the `data` folder within your project directory.

### Pre-trained Models

Pre-trained model weights are available for direct use:

🤗 **[Download Pre-trained Models](https://huggingface.co/liuzhijin/HDSE-MS-models/tree/main/HDSE-MS_pretrained)**

Available models:
- **NIST Model**: Trained on NIST23 dataset
- **MassBank Model**: Trained on MassBank dataset
- **All CE Model**: Trained on combined collision energy data

**Quick Download:**
```bash
# Install huggingface_hub
pip install huggingface_hub

# Download specific model (example: NIST model)
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='liuzhijin/HDSE-MS-models', filename='HDSE-MS_pretrained/nist_model/best_model_state.pth', local_dir='.')"
```

Or download manually from the link above and place the `.pth` files in your desired directory.

---

## Data Preparation and Usage

Follow these steps to prepare the data and run the pipeline:

1. **Prepare Dataset**  
   Obtain the MSP file for the target dataset (e.g., NIST23).

2. **Extract Raw Data**  
   Run the extraction script:
   ```bash
   python Prepare_scripts/parse_and_export_23.py
   ```

3. **Preprocess Data**  
   Run the preprocessing script:
   ```bash
   python Prepare_scripts/prepare_data.py
   ```
   > Data processing logic is adapted from [Roestlab/massformer](https://github.com/Roestlab/massformer).

4. **Train and Evaluate**  
   Execute training with the configuration file:  
   **Example (NIST23 Positive Ion Mode):**
   ```bash
   python src/train.py -c config/nist23_P.yml
   ```

---

## Molecular Attention Visualization

Visualize the attention weights of molecular structures to interpret model predictions.

### Setup Steps

1. **Uncomment Visualization Code**  
   In `src/model.py`, uncomment the code related to `if ((self.training))` to enable attention visualization during inference.

2. **Prepare Input Molecules**  
   Add SMILES strings of molecules you want to visualize in the `smiles_list`:
   ```python
   smiles_list = ["O=C(Nc1ccccc1)c1ccc(Cl)cc1"]
   ```
   > **Tip**: You can modify this part to load SMILES from CSV or TXT files for batch processing.

3. **Load Pre-trained Weights**  
   Modify the `embedder_prefix` and model path to load the appropriate pre-trained weights:
   ```python
   state_dict = torch.load('HDSE-MS_pretrained/nist_model/best_model_state.pth', map_location='cpu')
   embedder_prefix = "embedders.0."  # Adjust based on your model architecture
   ```

4. **Customize Visualization Parameters**  
   Modify parameters in the `draw_graph_with_attn` function to customize plot labels, colors, and layout:
   ```python
   draw_graph_with_attn(
       graph_data=batch_graph,
       attention_weights=attn_weights,
       title="Molecular Attention Heatmap",
       save_path="output/attention_viz.png"
   )
   ```

5. **Run Visualization**  
   Execute the model script:
   ```bash
   python src/model.py
   ```

### Example Code Snippet

```python
from HDSE_MS_data_utils import HDSE_MS_preprocess

def test_model():
    # Define molecules to visualize
    smiles_list = ["O=C(Nc1ccccc1)c1ccc(Cl)cc1"]
    data_list = []

    # Preprocess SMILES
    for idx, smiles in enumerate(smiles_list):
        data = HDSE_MS_preprocess(smiles, idx)
        data_list.append(data)

    batch_graph = Batch.from_data_list(data_list)

    # Load pre-trained model
    model = HDSE_MS_Embedder()
    state_dict = torch.load('HDSE-MS_pretrained/nist_model/best_model_state.pth', map_location='cpu')
    embedder_prefix = "embedders.0."
    embedder_state_dict = {
        k[len(embedder_prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(embedder_prefix)
    }
    
    result = model.load_state_dict(embedder_state_dict, strict=False)
    if len(result.missing_keys) == 0 and len(result.unexpected_keys) == 0:
        print("✓ Model weights loaded successfully!")
    else:
        print("⚠ Some weights failed to load:")
        print("  Missing keys:", result.missing_keys)
        print("  Unexpected keys:", result.unexpected_keys)

    # Generate predictions and visualizations
    try:
        output = model({"hdse": batch_graph})
        print("\n✓ Model output shape:", output.shape)
        print("✓ Attention visualization saved!")
    except Exception as e:
        print("\n✗ Error during model forward pass:")
        print(e)

global_plot_counter = 0
if __name__ == "__main__":
    test_model()
```

### Output

The script will generate:
- **Attention heatmaps**: Visualizing which atoms/bonds the model focuses on
- **Console output**: Model predictions and validation metrics

Visualization files will be saved in the specified output directory (default: `output/`).

### Tips

- **Batch Processing**: Load multiple SMILES from a file:
  ```python
  import pandas as pd
  df = pd.read_csv('molecules.csv')
  smiles_list = df['smiles'].tolist()
  ```

- **Different Models**: Change `embedder_prefix` for different model architectures:
  - NIST model: `"embedders.0."`
  - MassBank model: `"embedders.1."`
  - Custom models: Check your model's state dict keys

- **Customize Plots**: Adjust `draw_graph_with_attn` parameters for publication-quality figures




---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Data processing logic adapted from [Roestlab/massformer](https://github.com/Roestlab/massformer)


---

Feel free to modify this code based on your actual project structure or additional requirements.

