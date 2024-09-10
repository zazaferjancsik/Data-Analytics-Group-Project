import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

mmData = pd.read_csv("mammographic_masses_data.csv")


#################    Function   #################

def normalize(dataColumn):
    min_value = min(dataColumn)
    max_value = max(dataColumn)
    
    normalized_data = [(x-min_value)/(max_value - min_value) for x in dataColumn]

    return normalized_data





################    End of Functions    ##################

# mmData.describe()
# shows mean, median, standard deviation, minumum value, maximum value, 1st 2nd and 3rd quantile

# Sev1 = mmData.loc[mmData["Severity"]==1]
# print(Sev1["Margin"])

#Scatterplot between age and BA level, with color representing Severity level
# ageBA = px.scatter(mmData, x="Age", y="BA", color='Severity', title = 'BA level by Age')
# ageBA.show()

#Heatmap to show the correlation between shape and margin
# shapemargin = px.density_heatmap(mmData, x="Shape", y="Margin", title='Shape/Margin Heatmap')
# shapemargin.show()

#Cleaning Data
mmDataCL = mmData.copy()
mmDataCL = mmDataCL.dropna()

# # Plotting Cleaned Data
# ageBACL = px.scatter(mmDataCL, x="Age", y="BA", color='Severity', title = 'BA level by Age')
# ageBACL.show()

# shapemarginCL = px.density_heatmap(mmDataCL, x="Shape", y="Margin", title='Shape/Margin Heatmap')
# shapemarginCL.show()

#Normalizing Data
mmDataN = mmDataCL.copy()
mmDataN["Age"] = normalize(mmDataN['Age'])

# #Plotting Normalized Data
# ageBAN = px.scatter(mmDataN, x="Age", y="BA", color='Severity', title = 'BA level by Age')
# ageBAN.show()

# shapemarginN = px.density_heatmap(mmDataN, x="Shape", y="Margin", title='Shape/Margin Heatmap')
# shapemarginN.show()
#Normalization here makes sense before we feed the data to the computer, as machine learning functions can buidl models better



############## Task 4: Feature Engineering  ####################

#4.1

#Let severity be the target variable
#Now we build a model for this purpose

X = mmDataN.drop('Severity', axis=1) 
y = mmDataN['Severity']

#Train Random Forest Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

importances = model.feature_importances_

# Creating a DataFrame to hold the feature importances
feature_importances = pd.DataFrame(importances, index=X.columns, columns=['Importance'])

# print(feature_importances)

# TreeFig = px.bar(
#     feature_importances,
#     x='Importance',
#     y=feature_importances.index,
#     title='Feature Importances to predict Severity',
#     labels={'Importance': 'Importance Score', 'index': 'Features'},
#     )

# TreeFig.show()

#4.2

#Choose how many components to keep
pca = PCA(n_components=2)

pcaData = pd.DataFrame(pca.fit_transform(mmDataN), columns = ['PrincipalComponent1','PrincipalComponent2'] )

PCAfig = px.scatter(
    pcaData,
    x='PrincipalComponent1',
    y='PrincipalComponent2',
    title='PCA of Features',
)

PCAfig.show()