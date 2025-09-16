<!-- 1. 试试不带control. 只用test做预测
    1.1. RF, Unet, VIT, 传统颜色方法
    1.2. interpolation

2. 加上calibration
    2.1. 用 Control line 做normalization 
    2.2. 用 color calibration 做normalization
    2.3. 用 control line 加入作为input 调整 model structure -->


# Experimental Plan for LFA Stripe Concentration Prediction

## Objective

To evaluate various models and input strategies for predicting allergen concentration (ng level) from lateral flow assay (LFA) stripe images. The focus is on comparing traditional feature-based approaches with modern deep learning architectures, both with and without normalization using the control line.

---

## Phase 1: Predict Using Test Line Only

### 1.1. Baseline Models (No Control Line)

We will start by testing performance using only the **test line** as input. This sets a lower bound for model performance.

**Models to be tested:**
- ✅ Random Forest on handcrafted features (mean RGB, contrast, ratios, etc.)
- ✅ U-Net for dense image feature extraction
- ✅ Vision Transformer (ViT or DeiT Tiny)
- ✅ Traditional color-based regression or interpolation methods

Each model will be evaluated in both:
- Classification mode (e.g., classifying into ng buckets: 0, 0.1, 1, 2, ...)
- Regression mode (e.g., predicting the actual ng value)

### 1.2. Interpolation Approach

Using known ng levels (e.g., 0, 0.1, 1, 2, 4, 6, 8, 10), we will:
- Fit an interpolated curve (e.g., sigmoid, log scale)
- Use predicted intensity or class index to interpolate actual ng level

---

## Phase 2: Calibration and Normalization

Once the baseline is established, we will introduce **calibration strategies** to improve model robustness.

### 2.1. Normalization Using Control Line

The control line will be used to normalize test line features.

**Methods:**
- Feature-level normalization (e.g., `test_R / control_R`, histogram ratios)
- Normalize RGB channels independently
- Use control line mean color as a reference vector

### 2.2. Color Calibration Board (Optional)

- Illumination/white balance parameters
- Apply global color correction to all images

---

## Phase 3: Model Structure Modification Using Control Line

Here, the control line is **explicitly added as model input**.

### 3.1. Input Concatenation (Image Stacking)

[ test_line RGB ]
[ control_line RGB ]

- Stack the test and control stripe images vertically or horizontally
- Feed into a CNN/ViT as a single 2D input
- Model learns implicit normalization

### 3.2. Multi-Branch Model

test_img ──► CNN ──►
                    ─► concat ─► FC ─► output
ctrl_img ──► CNN ──►

- Two CNN branches for test and control line respectively
- Feature fusion before prediction head
- Supports better structural separation of roles

---

## Evaluation Metrics

- **Classification**: Accuracy, F1-score, Confusion Matrix
- **Regression**: MAE, MSE, R², Scatter plot of predicted vs true ng
- **Visualization**: Grad-CAM for CNN interpretability

---

## Notes

- All images will be preprocessed to a fixed size with slight cropping/padding.
- Data Augmentations may be introduced later.
- CSV file will store mapping from filenames to ng values, test/control type, and source (e.g., Cockroach, DerF).

---

