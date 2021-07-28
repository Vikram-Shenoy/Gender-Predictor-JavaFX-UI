
# Gender-Predictor-JavaFX-UI
### Explanation
### Prediction Graph
![Predicted Graph](https://user-images.githubusercontent.com/73354099/127266079-5977aecf-e9a5-4304-9fee-852834f3035f.jpeg)
#### The purple box shows the number of times it got it wrong, and the top left shows number of times it was female and it predicted female.
#### The bottom right shows number of times it was male, and that it got male prediction correctly.

>“True positive” for correctly predicted event values.

>“False positive” for incorrectly predicted event values.

>“True negative” for correctly predicted no-event values.

>“False negative” for incorrectly predicted no-event values.

### Receiver Operating Characteristic curve
![ROC Graph](https://user-images.githubusercontent.com/73354099/127266061-5f47baaa-c18f-45da-b37f-7d8882c4989e.jpeg)

#### AUC is the percentage of the ROC plot that is underneath the curve:
#### IMPORTANT: first argument is true values, second argument is predicted probabilities
print(metrics.roc_auc_score(dfy_test, y_pred_prob))
> The above function is used for the plot. More can be found in javaproj(1).py file.

## Setup
1. Download the entirety of the src folder.
2. Setup File path names in model.py and scene2controller.java
3. 2 Preloaded models have been provided filename.sav and filename2.sav one can use these or train their own model using javaproj(1).py 

# Authors
### Vikram Shenoy
### Vinayak K Prasad
