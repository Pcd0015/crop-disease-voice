After running training/train_disease_model.py on Colab, download these two
(or three) files and place them in this directory:

- crop_disease_model.tflite   (used by default — faster CPU inference)
- crop_disease_model.keras    (optional, full model — only needed if you
                                set use_tflite=False in disease_detection.py)
- class_names.json            (required — maps prediction indices to class names)

This directory is git-ignored for the actual model files (they're large)
— only this README and .gitkeep should be committed to version control.
