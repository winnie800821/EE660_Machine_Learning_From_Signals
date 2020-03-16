import numpy as np
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import r2_score
np.set_printoptions(threshold=np.inf)
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
from sklearn.ensemble import RandomForestRegressor

def main():
    pd.set_option('display.max_columns', None)
    from sklearn.neighbors import KNeighborsClassifier
    data = pd.read_csv('Life Expectancy Data.csv')
    # print(data.columns)
    data.rename(columns={'Life expectancy ': "Life_expectancy"}, inplace=True)
    data.rename(columns={'Adult Mortality': "Adult_Mortality"}, inplace=True)
    data.rename(columns={'infant deaths': "infant_deaths"}, inplace=True)
    data.rename(columns={'percentage expenditure': 'percentage_expenditure'}, inplace=True)
    data.rename(columns={'Hepatitis B': "Hepatitis_B"}, inplace=True)
    data.rename(columns={' BMI ': "BMI"}, inplace=True)
    data.rename(columns={'under-five deaths ': "under-five_deaths"}, inplace=True)
    data.rename(columns={'Total expenditure': "Total_expenditure"}, inplace=True)
    data.rename(columns={' HIV/AIDS': "HIV/AIDS"}, inplace=True)
    data.rename(columns={' thinness  1-19 years': "thinness_1-19_years"}, inplace=True)
    data.rename(columns={' thinness 5-9 years': "thinness_5-9_years"}, inplace=True)
    data.rename(columns={'Income composition of resources': "Income_composition_of_resources"}, inplace=True)
    data.rename(columns={'HIV/AIDS': "HIV_AIDS"}, inplace=True)
    data.rename(columns={'Measles ': "Measles"}, inplace=True)
    data.rename(columns={'Diphtheria ': "Diphtheria"}, inplace=True)

    # delet the data with null life expectancy value
    data['Life_expectancy'] = data['Life_expectancy'].fillna(999)
    drop_index = data[(data.Life_expectancy == 999)].index.tolist()
    data = data.drop(drop_index)

    # make life_expectancy as our output
    labels = data.loc[:, ['Life_expectancy']]
    # deal with categorical data "status" since it contains numerical quality
    data['Status'] = data['Status'].str.replace('Developed', '2', case=False)
    data['Status'] = data['Status'].str.replace('Developing', '1', case=False)


    # Separate the data to train, val and test data
    x, x_test, y, y_test = train_test_split(data, labels, test_size=0.2, train_size=0.8, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=50)

    ####fill the missing data with mean value
    x = x.fillna(x.mean())
    x_test = x_test.fillna(x.mean())

    x_num = x.loc[:, ['Year', 'Status', 'Life_expectancy', 'Adult_Mortality', 'infant_deaths', 'Alcohol',
                                  'percentage_expenditure', 'Hepatitis_B', 'Measles', 'BMI', 'under-five_deaths',
                                  'Polio', 'Total_expenditure', 'Diphtheria', 'HIV_AIDS', 'GDP', 'Population',
                                  'thinness_1-19_years', 'thinness_5-9_years', 'Income_composition_of_resources',
                                  'Schooling']]
    x_test_num = x_test.loc[:, ['Year', 'Status', 'Life_expectancy', 'Adult_Mortality', 'infant_deaths', 'Alcohol',
                                'percentage_expenditure', 'Hepatitis_B', 'Measles', 'BMI', 'under-five_deaths', 'Polio',
                                'Total_expenditure', 'Diphtheria', 'HIV_AIDS', 'GDP', 'Population',
                                'thinness_1-19_years', 'thinness_5-9_years', 'Income_composition_of_resources',
                                'Schooling']]
    # Since the highest correlation between country and life expectancy is 0.17, we decide not to use the feature "country"

    # standardize the data
    standar_all = preprocessing.StandardScaler().fit(x_num)
    x = standar_all.transform(x_num)
    x_test = standar_all.transform(x_test_num)
    x1 = pd.DataFrame(data=x, columns=x_num.columns)
    x_test1 = pd.DataFrame(data=x_test, columns=x_num.columns)
    corrmat = x1.corr()

    cols = abs(corrmat).nlargest(22, 'Life_expectancy')['Life_expectancy'].index
    related_col = cols.drop(['Life_expectancy']).drop(['Status']).drop(['Hepatitis_B']).drop(['infant_deaths']).drop(
        ['GDP']).drop(['Measles']).drop(['Population']).drop(['percentage_expenditure']).drop(['Diphtheria'])

    x = x1[related_col]
    x_test = x_test1[related_col]

    avg_ytest = np.mean(y_test)
    one_array = np.ones([len(y_test), 1])
    mean_arr = avg_ytest['Life_expectancy'] * one_array
    baseline_mse = mean_squared_error(y_test, mean_arr)
    baseline_err = 1 - r2_score(y_test, mean_arr)
    print('####################Baseline######################')
    print('The baseline for the test data:')
    print('MSE = ', baseline_mse)
    print('Error Rate=', baseline_err)
    print('###################Final model: Random Forest Regression########################')
    error_x = []
    error_test = []
    mse_x = []
    mse_test = []
    oob_error = []
    for j in range(10):
        pre_train, pre_X_train_pick, pre_y_train, pre_y_train_pick = train_test_split(x, y, test_size=1 / 3)
        RF = RandomForestRegressor(n_estimators=38, bootstrap=True, random_state=0, oob_score=True)
        RF.fit(pre_X_train_pick, pre_y_train_pick.values.ravel())
        predict_x = RF.predict(x)
        predict_test = RF.predict(x_test)
        acc_x = RF.score(x, y)
        acc_test = RF.score(x_test, y_test)
        error_x = np.append(error_x, 1 - acc_x)
        error_test = np.append(error_test, 1 - acc_test)
        oob_error = np.append(oob_error, 1 - RF.oob_score_)
        mse_x = np.append(mse_x, mean_squared_error(predict_x, y))
        mse_test = np.append(mse_test, mean_squared_error(predict_test, y_test))
    mean_oob_err = np.mean(oob_error)
    meanerror_x = np.mean(error_x)
    meanerror_test = np.mean(error_test)
    mean_mse_x = np.mean(mse_x)
    mean_mse_test = np.mean(mse_test)
    var_mse_test = np.var(mse_test)
    var_err_test = np.var(error_test)
    print('In our final model (38 trees):')
    print('In the whole training data')
    print('Mean MSE =', mean_mse_x)
    print('Mean Error Rate =', meanerror_x)
    print('In the test data')
    print('Mean MSE = %.3f with variance = %.3f ' % (mean_mse_test, var_mse_test))
    print('Mean Error Rate = %.5f with variance = %.5f ' % (meanerror_test, var_err_test))
    print('Out Of Sample Error = ', mean_oob_err)




if __name__ == "__main__":
    main()