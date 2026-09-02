# Historical experiment inventory

This document records what was inspected in the private `Projet7` repository and how it is represented publicly. It prevents the polished portfolio from erasing the breadth of the original research.

## Branch coverage

- `dev`: ten notebooks plus TensorBoard logs and experimental artifacts were inspected.
- `flask`: ten notebooks were inspected, including deployment-oriented copies and PyTorch/Transformers drafts.
- `main`: contained no substantial research implementation.

No raw dataset, pickle, trained model, log directory, local path or cloud identifier is copied into this repository.

## Notebook map

| Historical notebook | Evidence found | Public representation |
|---|---|---|
| `P7_notebook.ipynb` / `P7_notebook_test.ipynb` | broad EDA and project workflow; files too large for safe direct reuse | `01_data_preparation.ipynb` and this inventory |
| `test_mod_simple.ipynb` | CountVectorizer and TF-IDF logistic baselines; validation/test results | `02_classical_baseline.ipynb` and the README result table |
| `machine_learning_pipeline_mlfow.ipynb` | preprocessing, stratified CV, evaluation helpers, Random Forest run and MLflow | `02_classical_baseline.ipynb`, `06_mlflow_tracking.ipynb` |
| `tensorFlow_word2vec.ipynb` | skip-gram-style Word2Vec training, negative sampling, TensorBoard projector export | `03_word_embeddings.ipynb` |
| `clean_keras_sequential.ipynb` | Word2Vec/FastText matrices, LSTM/BiLSTM, Conv1D, model diagrams and TensorBoard | `03_word_embeddings.ipynb`, `04_sequence_model.ipynb` |
| `test_pipeline_machine_learnia.ipynb` | preprocessing pipeline, custom estimator, randomized hyperparameter search | `04_sequence_model.ipynb` |
| `TensorFlow_Ananlyse_Sentiment_inside_ML.ipynb` | small-BERT fine-tuning, five epochs, evaluation and example inference | `05_transformer.ipynb` |
| `Tensorflow_Classifieur_Binaire_Tweet.ipynb` | custom BERT subclass, AdamW/warm-up, label smoothing, TensorBoard and test metrics | `05_transformer.ipynb` |
| `elasticnet_mlflow_implementation.ipynb` | logistic grid search, five-fold CV, 75 fits, MLflow logging | `06_mlflow_tracking.ipynb` |
| `test_mlfow.ipynb` | compact MLflow connectivity/model logging experiment | `06_mlflow_tracking.ipynb` |
| `brouillon_keras_sequential.ipynb` | 75-cell research draft with five rendered plots | consolidated into embeddings/sequence studies, not copied verbatim |
| `brouillon_HG_BERT_Torch.ipynb` | early Hugging Face/PyTorch BERT exploration | transformer study, explicitly labelled exploratory |
| `test_model_avancé_1.ipynb` | incomplete advanced-model stub | omitted from performance claims |

## Preserved results

### Classical baselines

- Bag-of-Words logistic regression: validation accuracy **79.7151%**, test accuracy **79.4777%**.
- TF-IDF logistic regression: validation accuracy **79.5895%**, test accuracy **77.9415%**.
- A later classical pipeline recorded five-fold CV accuracy **0.699 ± 0.002** and F1 **0.722 ± 0.005**; the notebook evolved over time, so this result is not attributed to a final promoted model.

### Word embeddings

The TensorFlow Word2Vec notebook trained a 256-dimensional embedding with negative sampling for 20 epochs. Training accuracy rose from **0.2713** at epoch 1 to **0.8490** at epoch 4 while loss fell from **1.5574** to **0.6752**. These are embedding-training metrics, not downstream sentiment accuracy, so they are not mixed into the classifier comparison.

### Transformer experiments

- First small-BERT run: test accuracy **0.7613**, AUC **0.8200**, precision **0.7518**, recall **0.7789**, F1 **0.7651**.
- Custom BERT classifier: test accuracy **0.7781**, AUC **0.8295**, precision **0.7815**, recall **0.7863**, F1 **0.7839**.

### Sequence-model search

The randomized LSTM experiment recorded accuracy **0.6544**, macro-F1 **0.6409** and ROC-AUC **0.6545**. A separate Keras evaluation reported recall **0.9585** and precision **0.5440**, revealing a strongly shifted decision boundary. The notebook also displayed an inconsistent built-in accuracy value; therefore the public account reports the independently calculated metrics and flags the inconsistency instead of concealing it.

## Engineering lessons retained

1. Establish a strong sparse linear baseline before escalating model complexity.
2. Fit tokenizers, vectorizers and resampling steps only on training folds.
3. Compare accuracy, F1, ROC-AUC, precision and recall together.
4. Treat threshold selection and calibration as separate from representation learning.
5. Version preprocessing, model signatures and input examples with tracked experiments.
6. Keep historical results traceable, but make the public demo runnable without private artifacts.

