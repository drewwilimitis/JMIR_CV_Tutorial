# Cross-Validation Tutorial: Predictive Modeling in Healthcare
___

This open-source respository contains the applied demonstration of cross-validation methods for healthcare problems associated with our recent publication: *Practical Considerations and Applied Examples of Cross-Validation for Model Development and Evaluation in Health Care: Tutorial* (https://ai.jmir.org/2023/1/e49023). Our work outlines the funadmental approach of cross-validation for model evaluation and model selection with a focus on applied predictive modeling in healthcare. We offer an applied demonstration of cross-validation best practices with exemplar problems in classification (in-hospital morality prediction) and regression (length-of-stay prediction) using the widely available MIMIC-III dataset. <br>

We also list the following sources we found helpful.

Coding resources for implementing CV/Nested CV:<br>
1. https://github.com/casperbh96/Nested-Cross-Validation/blob/master/nested_cv/nested_cv.py [Python Version] <br>
2. https://github.com/stephenbates19/nestedcv_experiments [R Version] <br>
  - https://github.com/stephenbates19/nestedcv_experiments (numerical experiments for [2] sparse/low dimensional linear models) <br>

NOTE: for Number [**2**] - This method was introduced in the paper Cross-validation: what does it estimate and how well does it do it? by Stephen Bates, Trevor Hastie, and Robert Tibshirani, available at https://arxiv.org/abs/2104.00673


Resources for MIMIC/eICU: <br>
1. MIMIC/MIT Website with Docs/Code Links: https://mimic.mit.edu/ <br>
2. Code for building MIMIC Datasets: https://github.com/MIT-LCP/mimic-code (MIMIC-III or MIMIC-IV Ideal - Google BQ vs. Postgres)
3. https://github.com/clinicalml/ML-tools/tree/master/MIMIC-tools (Benchmarks & Modeling) <br>
4. https://github.com/MIT-LCP/mimic-workshop (Tutorial used for MIMIC workshop/bookcamp) <br>
5. https://github.com/GoogleCloudPlatform/healthcare/blob/master/datathon/mimic_eicu/tutorials/ (Google 
