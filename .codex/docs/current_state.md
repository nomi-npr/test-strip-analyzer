## Project Goal

Build a robust pipeline to analyze test strip images and determine **positive/negative results for 2 tests** per strip. The test works by comparing test line visibility against a control line - if the test line is less visible than the control, the result is **positive** (chemical was consumed).

**Dataset:** ~900 images, with ~60 positive results (class imbalance ~7%)

## Approaches to Test

### Approach A: Background Removal + CNN

Pipeline:
```
[Image] → [Remove Background using InSPyReNet] → [CNN] → [Test1: pos/neg, Test2: pos/neg]
```

### Approach B: End-to-End CNN Classification

Pipeline:
```
[Image] → [CNN] → [Test1: pos/neg, Test2: pos/neg]
```

---

## Background Removal Testing

### Tested Models

| Model | Status | Quality |
|-------|--------|---------|
| withoutBG | ❌ Rejected | Too bad |
| InSPyReNet (transparent-background) | ✅ Accepted | 95% accuracy |

### InSPyReNet Test Results

- **Processing time:** ~1.7 seconds per image
- **Success rate:** 95%

## Key Technical Decisions

1. **Background removal:** InSPyReNet via `transparent-background` library
2. **Class imbalance handling:** Focal Loss, weighted sampling, heavy augmentation

---

## Dependencies

```
# Core
opencv-python>=4.5.0
numpy>=1.20.0
Pillow>=9.0.0
scipy>=1.7.0

# Background removal
transparent-background>=1.2.0

# Deep learning (for Approach B)
torch>=1.9.0
torchvision>=0.10.0
onnxruntime>=1.10.0

# Training utilities
albumentations>=1.0.0
scikit-learn>=0.24.0
```