# Cross-Validation: Applied Tutorial for Predictive Modeling in Healthcare
___

We are releasing this open-source code base alongside a manuscript we hope to publish soon that covers cross-validation and model evaluation methods while offering an applied demonstration using best cross-validation practices using the widely available MIMIC-III dataset. <br>

As we await manuscript review and our code base requires further polishing, we give a list of sources below that we found helpful.


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
