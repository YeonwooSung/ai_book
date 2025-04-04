# Model Distillation
Model distillation refers to training an (often smaller) student model to mimic the behaviour of an (often larger) teacher model, or collection of teacher models. This is commonly used to make models **faster, cheaper and lighter**.

## Cross Encoder Knowledge Distillation
The goal is to minimize the difference between the student logits (a.k.a. raw model outputs) and the teacher logits on the same input pair (often a query-answer pair).

Here are two training scripts that use pre-computed logits from [Hostätter et al.](https://arxiv.org/abs/2010.02666), who trained an ensemble of 3 (large) models for the MS MARCO dataset and predicted the scores for various (query, passage)-pairs (50% positive, 50% negative).

- **[train_cross_encoder_kd_mse.py](train_cross_encoder_kd_mse.py)**
    ```{eval-rst}
    In this example, we use knowledge distillation with a small & fast model and learn the logits scores from the teacher ensemble. This yields performances comparable to large models, while being 18 times faster.

    It uses the :class:`~sentence_transformers.cross_encoder.losses.MSELoss` to minimize the distance between predicted student logits and precomputed teacher logits for (query, answer) pairs.
    ```
- **[train_cross_encoder_kd_margin_mse.py](train_cross_encoder_kd_margin_mse.py)**
    ```{eval-rst}
    This is the same setup as the previous script, but now using the :class:`~sentence_transformers.cross_encoder.losses.MarginMSELoss` as used in the aforementioned `Hostätter et al. <https://arxiv.org/abs/2010.02666>`_.

    :class:`~sentence_transformers.cross_encoder.losses.MarginMSELoss` does not work with (query, answer) pairs and a precomputed logit, but with (query, correct_answer, incorrect_answer) triplets and a precomputed logit that corresponds to ``teacher.predict([query, correct_answer]) - teacher.predict([query, incorrect_answer])``. In short, this precomputed logit is the *difference* between (query, correct_answer) and (query, incorrect_answer).
    ```
