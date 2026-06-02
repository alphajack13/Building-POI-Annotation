# BiSpaSem: Bidirectional Spatial-Semantic
Interaction for POI Category Annotation

These are the source codes for the **BiSpaSem** model and its corresponding data.

- Data
  
  The files in the datasets/ folder are the data used to train the **BiSpaSem** model.
  
  haidianF_data.csv is the data of HD
  
  lixiaF_data.csv is the data of LX
  
  NY/poi_2024_with_cate.csv is the data of BK
  
  TYO/poi_2024_with_cate.csv is the data of TKY
  
  The data in the file is random. We take the first 80% (sorted by their subscripts in ascending order) as the training set and the last 20% as the test set. When considering the neighbors of a POI, we ignore any POIs that belong to the test set. Note that the format of longitude and latitude coordinates in HD and LX is GCJ-02.
  
- Code

  1. src/models/GeoSemPA.py is the code of the main structure of the **BiSpaSem** model.
  
  2. src/models/TCSE_model is the code of **Text-centric Contextual Semantic Encoder**.
  
  3. src/main.py is executed to train the model.
  
  4. Base semantic model see: 
  
     https://huggingface.co/google-bert/bert-base-chinese
  
     https://huggingface.co/google-bert/bert-base-uncased
  
     https://huggingface.co/tohoku-nlp/bert-base-japanese-v3
  
  5. The model weights of the pre-trained Text-centric Contextual Semantic Encoder can be accessed at https://drive.google.com/drive/folders/10z05U8lMkvB41JtLKf-sRTVlm3yhPL2q?usp=drive_link
  

