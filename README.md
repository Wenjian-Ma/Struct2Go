# Code for paper "Is the AlphaFold Predicted Protein Structure Comparable to the Experimentally Solved Structure in Predicting Protein Function? A Comparative Study"
Dependencies
---

python == 3.7.11

pytorch == 1.7.1

PyG (torch-geometric) == 2.0.2

numpy == 1.21.2

Data preparation
---
Experimental solved protein structures and AlphaFold predicted structures are available at [LinkA](https://pan.baidu.com/s/1p6E2UuLaih1Ehs59s8VbbQ "password:1234") and [LinkB](https://pan.baidu.com/s/1_mNskJfLNL9AiOZrDd71cA "password:1234"), respectively.

Unzip **contact_map_dense_alphafold_8.0.tar.gz**, **contact_map_dense_pdb_8.0.tar.gz** and **contact_map_dense_pdb_8.0_for_external_test.tar.gz** to **Struct2Go/data_collect/**

Model preparation
---
1. Trained models for MF-, BP-, CC-GO terms prediction are available at [LinkC](https://pan.baidu.com/s/1tNw8KxH6lhsUX4ATSW5wtQ "password:1234"), [LinkD](https://pan.baidu.com/s/1o4wauLflll75EkKopBY5Wg "password:1234"), [LinkE](https://pan.baidu.com/s/1kNR-OTnGDxFNOvcEJb4iSA "password:1234"), respectively.

   Unzip these model files(.pkl) downloaded above to **Struct2Go/data_collect/mf/model/**, **Struct2Go/data_collect/bp/model/** and **Struct2Go/data_collect/cc/model/**, respectively.

2. Pre-trained Bi-LSTM language model is available at [Link](https://pan.baidu.com/s/1nTWUk4KeqXhnskRMq2Bm0A "password:1234").
   
   Unzip the model file to **Struct2Go/saved_models/**.

Train
---
python main.py

Test
---
python test.py

Test for external validation set
---
python external_test.py


