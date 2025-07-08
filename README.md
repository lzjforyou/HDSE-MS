## Data Preparation and Usage

Follow these steps to prepare the data and run the pipeline:

1.  **Prepare Dataset**  
    Obtain the MSP file for the target dataset (e.g., NIST23).

2.  **Extract Raw Data**  
    Run the extraction script:
    ```bash
    python Prepare_scripts/parse_and_export_23.py
    ```

3.  **Preprocess Data**  
    Run the preprocessing script:
    ```bash
    python Prepare_scripts/prepare_data.py
    ```
    > Data processing logic is adapted from [Roestlab/massformer](https://github.com/Roestlab/massformer).

4.  **Train and Evaluate**  
    Execute training with configuration file:  
    **Example (NIST23 Positive Ion Mode)**:
    ```bash
    python src/train.py -c config/nist23_P.yml
    ```
