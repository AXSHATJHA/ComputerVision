import json

with open("implementation.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        
        # fix Keras imports
        source = source.replace("from keras.layers import", "from tensorflow.keras.layers import")
        source = source.replace("from keras.preprocessing.image import", "from tensorflow.keras.preprocessing.image import")
        source = source.replace("from tensorflow.keras.layers.experimental import preprocessing\n", "")
        source = source.replace("layers.experimental.preprocessing.", "layers.")
        
        # fix missing ')' at the end of test_images cell due to previous mistake
        if "test_images = generator.flow_from_dataframe(" in source and ")\n" not in source[-5:] and ")" not in source[-5:]:
            source = source + "\n)"
            
        # fix path
        if 'path = kagglehub.dataset_download(' in source and 'path = os.path.join(path, "animals", "animals")' not in source:
            source = source.replace('path = kagglehub.dataset_download("iamsouravbanerjee/animal-image-dataset-90-different-animals")\n', 
                                    'path = kagglehub.dataset_download("iamsouravbanerjee/animal-image-dataset-90-different-animals")\nimport os\npath = os.path.join(path, "animals", "animals")\n')
            
        # Instead of split/join which could truncate, let's use list formatting
        # keepends=True works properly
        cell["source"] = source.splitlines(keepends=True)

with open("implementation.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
