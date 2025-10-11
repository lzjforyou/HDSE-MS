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

   **Or use pre-trained models for inference:**
   ```bash
   python src/inference.py -c config/nist23_P.yml --model_path HDSE-MS_pretrained/nist_model/best_model_state.pth
   ```


---

Feel free to modify this section based on your actual project structure or additional requirements.

