# AnemiaScan - anemia screening aid from an inner-eyelid photo
ANN project (scikit-learn MLP + Streamlit). **Screening aid, not a diagnosis.**

## Pipeline
1. **Segmentation** (`segment.py`): isolate the eyelid (OpenCV: Lab 'a' channel + Otsu + largest blob).
2. **Feature engineering** (`features.py`): mean/std of RGB, HSV, Lab over eyelid pixels + red ratios (21 features).
3. **Scaling**: StandardScaler.  4. **Model**: `MLPClassifier` (ANN), small network + L2 regularization (alpha) to limit overfitting; RandomForest as baseline.
5. **Split**: train/test split by *source image group* so augmented copies never leak into the test set.

## Run
    pip install -r requirements.txt
    python train.py "path/to/dataset_folder"     # creates anemia_model.joblib, metrics.json, confusion_matrix.png
    streamlit run app.py

## Dataset notes (check licence before submitting/publishing!)
Ghana conjunctiva dataset (anemic / non-anemic folders by filename). Images are pre-segmented crops.
4262 files -> 2015 unique after removing exact duplicates -> ~520 source eyelids. Labels are binary: no haemoglobin values.
