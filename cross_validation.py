
import numpy as np
import statsmodels.api as sm
import pandas as pd
from sklearn.model_selection import KFold


loansData = pd.read_csv('https://github.com/Thinkful-Ed/curric-data-001-data-sets/raw/master/loans/loansData.csv')

cleanFICORange = loansData['FICO.Range'].map(lambda x: x.split('-'))

cleanFICORange = cleanFICORange.map(lambda x: [int(n) for n in x])

loansData['FICO.Range'] = cleanFICORange

fico_score = [i[0] for i in loansData['FICO.Range']]

loansData['FICO.Score'] = fico_score

cleanInterestRate = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))
cleanLoanLength = loansData['Loan.Length'].map(lambda x: int(x.rstrip(' months')))

loansData['Loan.Length'] = cleanLoanLength
loansData['Interest.Rate'] = cleanInterestRate
         
intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

# The dependent variable
y = np.matrix(intrate).transpose()
# The independent variables shaped as columns
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()
    
x = np.column_stack([x1,x2])

X = sm.add_constant(x)

model = sm.OLS(y,X)
f = model.fit()

#Split data into 10 sets
kf = KFold(n_splits=10)

for train, test in kf.split(X):
    print("%s %s" % (train, test))

