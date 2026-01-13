# Paper Implementation Plan (Frontiers in AI, 2023)

This plan adapts the paper’s end-to-end methodology to our strip dataset and current codebase. It is structured to (1) reproduce the paper’s core findings as faithfully as possible, and (2) extend them to our multi-label use case (T1/T2) without losing comparability.

Paper: "A deep learning-based approach for lateral flow assay test strip recognition" (Frontiers in AI, 2023).  
Key ideas: baseline CNNs -> transfer learning -> StyleGAN2-ADA for data augmentation -> evaluate on real/synthetic images.  

Paper link: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1235204/full
Paper code sample link: https://www.kaggle.com/code/letitiacowell/covid19-lateral-flow-from-image-classifier#Gap-1:-Cross-Domain-Data-Heterogeneity-and-Transfer-Learning-Robustness

---

## 0) Goals, Scope, and Success Criteria

**Primary goals**
- Replicate the paper’s core methodology on our dataset with minimal deviations.
- Provide an apples-to-apples benchmark for binary positive/negative classification.
- Extend to our required multi-label task (T1/T2) to keep product relevance.

**Success criteria**
- Reproduce the paper’s evaluation ladder:
  - Real test set performance (baseline).
  - Synthetic-only test performance.
- Clear comparison between baseline CNNs, transfer learning, and GAN-augmented training.
- Documented best model(s) for binary and multi-label tasks.

---

## 1) Dataset Mapping and Label Strategy

**Paper task:** binary positive vs negative.  
**Our task:** multi-label (T1, T2), with "positive if test < control".

**Multi-label task (product-aligned):**
   - Two binary outputs: T1 positive, T2 positive.
   - This retains our current ground truth format.

**Deliverable:** Two label CSVs (or one with derived columns), stored in `data/splits/`:
- `train_multilabel.csv`, `val_multilabel.csv`, `test_multilabel.csv`

---

## 2) Dataset Curation and Splits

The paper evaluates on two kinds of sets: **real**, **synthetic**.

We will maintain:
1) **Real dataset**: current curated set.
2) **Synthetic dataset**: generated with StyleGAN2-ADA per class (positive/negative).

**Split strategy (paper-style):**
- 80% train / 20% val for model selection.
- Held-out test set (real).
- Separate synthetic test set.

---

## 3) Preprocessing, Input Resolution and Augmentation

The paper explores multiple input sizes (224, 200, 128). We will match that.

**Preprocessing**
- Normalize to [0, 1] (or mean/std if needed).
- Keep **raw images** as the primary path, to match paper’s "uncontrolled background".
- Optionally run the same models with our **bg_removed** variant for internal comparison.

**Input sizes**
- 224x224, 200x200, 128x128
- For each model.

Citation from paper:

"Several data preprocessing approaches such as outlier detection, data scaling, and data transformation are utilized for this research. Experiments were performed involving different dimensions of the images to analyse the most accurate model. The three different dimensions used in these experiments include 224 × 224 × 3, 200 × 200 × 3, and 128 × 128 × 3 for the first, second, and third experiments, respectively. All the images have undergone a two-step verification process to confirm the class to which it is added. Figure 2 show samples from the two classes after labeling the dataset. The images were scaled and normalized before training the models. The image pixel values are normalized between 0 and 1, and the image sizes scaled to different standard sizes as mentioned above."

**Augmentation**
As we have way less positives than negatives, we'll use the following data augmentation strategies: random rotation (±10°), translation (±10 pixels), and brightness adjustment (±20%). We need the final training dataset to have X negative images, X/3 T1 positives, X/3 T2 positives and X/6 T1 and T2 positives. So augment as necessary.

---

## 4) Baseline CNN (Paper’s "Vanilla CNN")

Implement the paper’s vanilla CNN architecture:
- 3 Conv blocks: filters [16, 32, 64], kernel 3x3
- MaxPool 2x2 after each conv block
- Dropout 0.2
- Dense 256
- Output: 2-class softmax for binary

Train with both:
- **Adam** and **SGD (momentum=0.9)**
- Hyperparameters from the paper’s Table 2, see below:

| Hyperparameter       | Range/Values |
| -------------------- | ------------ |
| Number of CNN layers | 3            |
| CNN filter sizes     | 16, 32, 64   |
| Dropout rate         | 0.2          |
| Learning rate        | 0.1 - 0.0001 |
| Beta_1               | 0.5 - 0.9    |
| Beta_2               | 0.7 - 0.999  |
| Epsilon              | 1e-08        |

The network using Stochastic Gradient Descent as an optimisation algorithm further used a momentum parameter value of 0.9 after initial empirical tests.

**Deliverables**
- Configurable baseline model in `training/models/` or equivalent.
- Training scripts to run baseline at 224/200/128.
- Metrics logged to `training/runs/...`

---

## 5) Transfer Learning Models (Paper Replication)

Paper uses NASNetMobile, DenseNet121, ResNet50.

**Primary replication**
- NASNetMobile
- DenseNet121
- ResNet50

**Training procedure**
- Freeze backbone; train head (dense + BN + ReLU + softmax).
- Same optimizer and hyperparameter ranges as baseline.

**Deliverables**
- Transfer learning training runs for all input sizes.
- Best model per input size and task.

Citation from paper:

"Transfer learning is a machine learning method that uses a pre-trained model mainly as the feature extraction layer. Using the weights of the convolutional layers belonging to the pre-trained model can reduce the number of network parameters to be trained (Kaur and Gandhi, 2020). When this technique is used, only the final dense layers are (re)trained with the input data. In this study, three distinct pre-trained models have been adopted to find the best-performing model in comparison with the default Vanilla CNN models. They are NASNetMobile, DenseNet121, and ResNet50. These models are basically CNNs pre-trained with the ImageNet database that contains more than a million images (Deng et al., 2009). The principal advantages of using transfer learning are resource conservation, along with enhanced efficiency while training the new models with fewer training samples. Each pre-trained network was appended with custom dense layers to redesign the output layer according to the dataset. The dense layer contains a batch normalization technique to reduce the overfitting of the model. Similar to the Vanilla CNN model, ReLU was used as the activation function and Softmax as the output layer activation function. All the pre-trained models used Adam as the optimisation algorithm and same hyperparameters (as shown in the Table 2) for Vanilla CNN is also tuned for these models. Ultimately, the most accurate learning model was chosen depending on the best validation accuracy and loss. The train and validation data split was 80%–20% with a training dataset of 640 images and a validation dataset of 160 images."
---

## 6) StyleGAN2-ADA Augmentation

The paper trains StyleGAN2-ADA per class and augments training.

**Steps**
1) Train StyleGAN2-ADA on **positive** samples.
2) Train StyleGAN2-ADA on **negative** samples.
3) Generate synthetic images (start 100/100 to match paper).
4) Train "Model DR" = best transfer model retrained on real + synthetic.

**Deliverables**
- Synthetic dataset folders:
  - `data/synthetic/positive/`
  - `data/synthetic/negative/`
- Updated training pipeline to include synthetic data.

---

## 7) Evaluation Protocol

Metrics to report (paper-style):
- Macro Accuracy, precision, recall, F1
- Per-label precision/recall/F1
- Joint metrics (both labels correct).
- Confusion matrices
- Separate evaluations on:
  - Real test set
  - Synthetic-only test set

**Statistical testing**
- 10-fold cross-validation
- Paired t-test on accuracy

---

## 8) Result Aggregation and Reporting

Create a structured results table to match the paper:
- Model vs input size vs dataset split (real/synth)
- Baseline CNN vs transfer vs transfer+GAN

**Deliverables**
- `training/runs/paper_results_summary.json`
- A markdown summary in `reports/` or `.codex/`

---

## 9) Repo Integration Plan

**Where to implement**
- Models: `training/models/`
- Training entrypoint: `training/train.py`
- Orchestration: `scripts/run_full_experiments.py`
- GAN training scripts: new in `training/models/`
- Evaluation aggregation: `scripts/summarize_paper_results.py`

**New config entries**
- `approach`: paper_multilabel
- `input_size`: 224 / 200 / 128
- `dataset_split`: real / synthetic

---

## 10) Timeline and Execution Order

**Phase 1: Baselines**
- Implement vanilla CNN
- Run 3 input sizes x 2 optimizers
- Record metrics

**Phase 2: Transfer Learning**
- Run NASNetMobile, DenseNet121, ResNet50
- 3 input sizes x 2 tasks

**Phase 3: StyleGAN2-ADA**
- Train GANs
- Generate synthetic images
- Retrain best model (Model DR)

**Phase 4: Evaluation + report**
- Real vs synthetic performance
- Final summary + recommended model

---

## 11) Acceptance Checklist

- [ ] Baseline CNN reproduced (3 input sizes, 2 optimizers)
- [ ] Transfer learning reproduced (NASNet/DenseNet/ResNet)
- [ ] StyleGAN2-ADA synthetic data generated and integrated
- [ ] Real vs synthetic evaluation completed
- [ ] Binary and multi-label results reported
- [ ] Paper-aligned comparison table produced
