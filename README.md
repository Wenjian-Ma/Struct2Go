Code for paper "Enhancing protein function prediction performance by utilizing AlphaFold-predicted protein structures"
---

Dependencies
---

python == 3.7.11

pytorch == 1.7.1

PyG (torch-geometric) == 2.0.2

numpy == 1.21.2

Data preparation
---
1. Unzip ''**contact_map_dense_alphafold_8.0.tar.gz**'', ''**contact_map_dense_pdb_8.0.tar.gz**'' and ''**contact_map_dense_pdb_8.0_for_external_test.tar.gz**'' to **Struct2Go/data_collect/**
2. Unzip ''**contact_map_6582_8.0.tar.gz**'' to **Struct2Go/data_collect/amplify_samples/**

Model preparation
---
**1. 30% Sequence Identity-Based:**

1). **_PDB-A_ vs _Expanded_**: Trained models for MF-, BP, CC-GO terms prediction are available at [Link](https://pan.baidu.com/s/1_XuQQnKkIJbAp0HvoTH02w "password:1234")

   Unzip this .rar file to **/data_collect/amplify_samples/model**

2). **_PDB-B_ vs _AF_**: Trained models for MF-, BP-, CC-GO terms prediction are available at [Link](https://pan.baidu.com/s/1C_3jdvLxkBoQyErC4NN-jQ "password:1234")

   Unzip this .rar file to **/data_collect/**
   
**2. Random Splitting-Based:**

1). **_PDB-A_ vs _Expanded_**: Trained models for MF-, BP, CC-GO terms prediction are available at [Link](https://pan.baidu.com/s/1S76i0rgWDBfymcKeP3uD4A "password:1234")

   Unzip this .rar file to **/data_collect/amplify_samples**

2). **_PDB-B_ vs _AF_**: Trained models for MF-, BP-, CC-GO terms prediction are available at [LinkA](https://pan.baidu.com/s/1tNw8KxH6lhsUX4ATSW5wtQ "password:1234"), [LinkB](https://pan.baidu.com/s/1o4wauLflll75EkKopBY5Wg "password:1234"), [LinkC](https://pan.baidu.com/s/1kNR-OTnGDxFNOvcEJb4iSA "password:1234"), respectively.

   Unzip these model files(.pkl) downloaded above to **Struct2Go/data_collect/mf/model/**, **Struct2Go/data_collect/bp/model/** and **Struct2Go/data_collect/cc/model/**, respectively.

**3. Pre-trained Protein Language Model:**

   Pre-trained Bi-LSTM language model is available at [Link](https://pan.baidu.com/s/1nTWUk4KeqXhnskRMq2Bm0A "password:1234").
   
   Unzip the model file to **Struct2Go/saved_models/**.

Test
---
**1. 30% Sequence Identity-Based:**

**_PDB-A_ vs _Expanded_**: python main_amplified_samples_test_30.py

**_PDB-B_ vs _AF_**: python test_7198.py

**2. Random Splitting-Based:**

**_PDB-A_ vs _Expanded_**: python main_amplified_samples_test.py

**_PDB-B_ vs _AF_**: python test.py

Test for external validation set
---
python external_test.py

Train
---
**1. 30% Sequence Identity-Based:**

**_PDB-A_ vs _Expanded_**: python main_amplified_samples_30.py

**_PDB-B_ vs _AF_**: python main_7198.py

**2. Random Splitting-Based:**

**_PDB-A_ vs _Expanded_**: python main_amplified_samples.py

**_PDB-B_ vs _AF_**: python main.py

If you need pdb files we collectd:
---
Experimental solved protein structures (PDB-A) are available at [Link](https://pan.baidu.com/s/1DrzpvFDT-_dHsRYPuG7EpA "password:1234")

Experimental solved protein structures (PDB-B) and AlphaFold predicted structures (AF) are available at [LinkE](https://pan.baidu.com/s/1p6E2UuLaih1Ehs59s8VbbQ "password:1234") and [LinkF](https://pan.baidu.com/s/1_mNskJfLNL9AiOZrDd71cA "password:1234"), respectively.

