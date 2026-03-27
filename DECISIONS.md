# Decision Log

## 1. **Data Preprocessing Strategy**
   I used a simple median imputation for missing numerical features and mode imputation for missing categorical features to prevent data leakage and handle potential missing values cleanly. I also utilized one-hot encoding for categorical variables to make them compatible with the linear layers of our Multilayer Perceptron. 

## 2. **Model Architecture and handling class imbalance**
   I implemented a 3-layer Multilayer Perceptron (MLP) with ReLU activations and Dropout (0.3). Given the relatively small dataset size (3,800 records), an extremely deep network would likely overfit. The dropout layers act as regularization. I used Binary Cross Entropy with Logits Loss (BCEWithLogitsLoss) for final binary classification.

   ### Why I did it:
   The dataset is heavily imbalanced (91% vs 9% positives). The rising loss still indicated under-learning of minority patterns, so I increased sensitivity using pos_weight=10. A small MLP was chosen to avoid overfitting on a relatively small dataset with high-dimensional one-hot features.

   ### What I considered and rejected:
   I considered class-balanced sampling but rejected them because they can distort real clinical distributions. I also considered tree-based models (XGBoost), but focused on a deployable deep learning pipeline requirement.

   ### What would happen if I was wrong here:
   If class weighting is too aggressive, the model may overpredict positives, increasing false positives and reducing clinical usefulness.


## 3. **Evaluation metric and threshold selection**

   ### What I did:
   I used AUROC, precision, recall, and F1-score, and selected the decision threshold by maximizing F1 over values from 0.1 to 0.85. The best threshold was 0.55, producing F1 = 0.18.

   ### Why I did it:
   Because the dataset is highly imbalanced, accuracy would be misleading. The model achieved very high recall (~0.82) but extremely low precision (~0.102), showing it predicts many positives. In a medical readmission context, missing a readmitted patient is costly, so I prioritized recall-informed F1 optimization rather than fixed 0.5 threshold.

   ### What I considered and rejected:
   I considered using AUROC as the main decision metric but rejected it because it ignores threshold behavior. I also considered maximizing precision but it would have severely reduced recall.

   ### What would happen if I was wrong here:
   If threshold tuning is misaligned, the model could either miss high-risk patients (false negatives) or overload the system with false alerts (false positives).