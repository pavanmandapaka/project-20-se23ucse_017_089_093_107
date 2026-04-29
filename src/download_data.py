# src/download_data.py
from multiversity.multicare_dataset import MedicalDatasetCreator

def main():
    print("Initializing MedicalDatasetCreator...")
    # This will pull files from Zenodo into your data/ folder
    mdc = MedicalDatasetCreator(directory='data')
    
    print("Setting up filters for MRI Brain Tumor cases...")
    filters = [
        {'field': 'case_strings', 'string_list': ['tumor', 'cancer', 'carcinoma'], 'operator': 'any'},
        {'field': 'caption', 'string_list': ['metastasis', 'tumor', 'mass'], 'operator': 'any'},
        {'field': 'label', 'string_list': ['mri', 'head']}
    ]
    
    print("Generating dataset subset...")
    # Creates a highly specific multimodal dataset for Pranav to test
    mdc.create_dataset(
        dataset_name='vlm_mri_subset', 
        filter_list=filters, 
        dataset_type='multimodal'
    )
    print("Data pipeline complete. Dataset ready in data/ folder.")

if __name__ == "__main__":
    main()