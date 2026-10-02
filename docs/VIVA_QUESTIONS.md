# Viva Questions and Answers

1. **What is churn?**  
Customer discontinuation of the telecom service.

2. **What type of ML problem?**  
Binary classification.

3. **What is the target?**  
`Churn`, encoded as 1 for Yes and 0 for No.

4. **Why remove customerID?**  
It is an identifier, not a meaningful predictive feature.

5. **Why convert TotalCharges?**  
It is stored as text in the raw data and contains blank entries.

6. **How are missing values handled?**  
Median imputation for numeric features inside the preprocessing pipeline.

7. **Why one-hot encode?**  
Most ML algorithms require numerical feature representations.

8. **Why scale numeric features?**  
To put numerical variables on comparable scales, especially useful for distance-based KNN and linear models.

9. **Why stratify the train/test split?**  
To preserve the churn/non-churn class proportion.

10. **Why Recall?**  
It measures the proportion of actual churners detected and directly reflects the cost of missed churners.

11. **Recall formula?**  
TP / (TP + FN).

12. **Precision formula?**  
TP / (TP + FP).

13. **What is F1-score?**  
The harmonic mean of precision and recall.

14. **What is a confusion matrix?**  
A table of TP, TN, FP and FN.

15. **Why K-fold CV?**  
To evaluate model stability across multiple splits.

16. **Why Stratified K-fold?**  
To preserve class proportions in each fold.

17. **What is Logistic Regression?**  
A linear probabilistic classification algorithm.

18. **What is KNN?**  
A distance-based classifier using nearby training examples.

19. **What is a Decision Tree?**  
A tree of learned feature-based decision rules.

20. **What is Random Forest?**  
An ensemble of many decision trees.

21. **What is Gradient Boosting?**  
Sequentially built weak learners where later learners focus on previous errors.

22. **Does high feature importance prove causation?**  
No. It shows predictive association, not causal effect.

23. **What does the web app probability mean?**  
The model's estimated probability of the churn class, not a guarantee.

24. **Why can the model be wrong?**  
Customer behavior changes, data may be incomplete, and historical patterns may not generalize.

25. **Can this be directly deployed in a telecom company?**  
It is a prototype. Production use requires real data, validation, monitoring, calibration, privacy and governance.
