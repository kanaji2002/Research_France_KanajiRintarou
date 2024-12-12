# Research_France_KanjiRintarou


![De_Normalized](Excute_2process/Result/Not_Normalized.png)

I excuted PyCaret libraly and got this figure about MAE score.

Feature is below. And target is total power.
- layers_1 (Number of layer in model1)
- layers_2
- batch_size_1
- batch_size_2
- model1_Param
- model2_Param
- model1_FLP
- model2_FLP
And I understand Linear Regression is good score.
Please tell me this feature and result is good or not. 



![Normalized](Excute_2process/Result/Normalized.png)
This is the result of normalizing Power and regression forecasting. After regression forecasting, the results were denormarlized. I used MAE as the evaluation method.

Feature is below(Same with above). And target is total power.
- layers_1 (Number of layer in model1)
- layers_2
- batch_size_1
- batch_size_2
- model1_Param
- model2_Param
- model1_FLP
- model2_FLP

Lasso and LassoLars had the lowest value at 4.85. However, when the experiment was performed with the same variables without normalization, Lenear Regressor was 4.75.

