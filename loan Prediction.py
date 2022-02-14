
import pyforest
df=pd.read_csv(r"C:/Users/hp/Downloads/loan_data_set.csv")
df.head()
df.info()
df.columns
df.shape
df.Loan_Status.value_counts()
df.describe().T
df.nunique()
df.Gender.replace(['Male','Female'],[0,1],inplace=True)
df.Married.replace(['No','Yes'],[0,1],inplace=True)
df.Dependents.replace(['0','1','2','3+'],[0,1,2,3],inplace=True)
df.Education.replace(['Graduate','Not Graduate'],[0,1],inplace=True)
df.Self_Employed.replace(['No','Yes'],[0,1],inplace=True)
df.Property_Area.replace(['Semiurban','Urban','Rural'],[0,1,2],inplace=True)
df['Gender']=df.Gender.fillna(df.Gender.mode()[0])
df.Married.fillna(df.Married.mode()[0],inplace=True)
df.Dependents.fillna(df.Dependents.mode()[0],inplace=True)
df.Education.fillna(df.Education.mode()[0],inplace=True)
df.Self_Employed.fillna(df.Self_Employed.mode()[0],inplace=True)
df.Credit_History.fillna(df.Credit_History.mode()[0],inplace=True)
df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mode()[0],inplace=True)
df.LoanAmount.fillna(df.LoanAmount.mode()[0],inplace=True)
df.isnull().sum()
df.Loan_Amount_Term.value_counts()
df.Loan_Status.value_counts()
df.Loan_Status.replace(['Y','N'],[1,0],inplace=True)

X=df.iloc[:,1:12]
y=df.iloc[:,12]
classifier = ('Gradient Boosting','Random Forest','Decision Tree','K-Nearest Neighbor','SVM')
y_pos = np.arange(len(classifier))
score = []
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier()
scores = cross_val_score(clf, X, y,cv=5)
score.append(scores.mean())
print('The accuration of classification is %.2f%%' %(scores.mean()*100))

from sklearn.ensemble import RandomForestClassifier
clf2=RandomForestClassifier()
scores=cross_val_score(clf2,X,y,cv=5)
score.append(scores.mean())
print('The accuration of classification is %.2f%%' %(scores.mean()*100))


