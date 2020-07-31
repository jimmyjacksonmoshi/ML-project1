import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ClassificationReport
url = "https://www.cameronvetter.com/wp-content/uploads/2018/07/WA_Fn-UseC_-Sales-Win-Loss.csv"
pd.set_option('max_columns', None)
sales_data = pd.read_csv(url)
print(sales_data.head(n=2))
print(sales_data.tail(n=2))
sns.set(style="whitegrid", color_codes=True)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot('Route To Market',data=sales_data,hue = 'Opportunity Result')
sns.despine(offset=10, trim=True)
plt.show()
sns.set(rc={'figure.figsize':(16.7,13.27)})
sns.violinplot(x="Opportunity Result",y="Client Size By Revenue", hue="Opportunity Result", data=sales_data);
plt.show()

le = preprocessing.LabelEncoder()
sales_data['Supplies Subgroup'] = le.fit_transform(sales_data['Supplies Subgroup'])
sales_data['Region'] = le.fit_transform(sales_data['Region'])
sales_data['Route To Market'] = le.fit_transform(sales_data['Route To Market'])
sales_data['Opportunity Result'] = le.fit_transform(sales_data['Opportunity Result'])
sales_data['Competitor Type'] = le.fit_transform(sales_data['Competitor Type'])
sales_data['Supplies Group'] = le.fit_transform(sales_data['Supplies Group'])

print(sales_data.head())

cols = [col for col in sales_data.columns if col not in ['Opportunity Number','Opportunity Result']]
data = sales_data[cols]
target = sales_data['Opportunity Result']
print(data.head(n=2))
print(target.head(n=2))

data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 10)
svc_model = LinearSVC(random_state=0)
pred = svc_model.fit(data_train, target_train).predict(data_test)
print("LinearSVC accuracy : ",accuracy_score(target_test, pred, normalize = True))

visualizer = ClassificationReport(svc_model, classes=['Won','Loss'])
visualizer.fit(data_train, target_train)
visualizer.score(data_test, target_test)
g = visualizer.show()
