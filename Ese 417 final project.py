import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = data.columns
    vif_values = []

    for i in range(data.shape[1]):
        y = data.iloc[:, i]
        X = data.drop(data.columns[i], axis=1)
        model = LinearRegression()
        model.fit(X, y)
        r_squared = model.score(X, y)
        # calculate
        vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
        vif_values.append(vif)
    vif_data["VIF"] = vif_values
    return vif_data

#remove features with high VIF
def remove_high_vif(df, drop_columns=None, threshold=10):
    if drop_columns is None:
        drop_columns = []
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.difference(drop_columns)
    data = df[numeric_features]
    removed_features = []
    while True:
        vif_df = calculate_vif(data)
        max_vif = vif_df["VIF"].max()
        if max_vif <= threshold:
            break
        # find maximum
        max_vif_feature = vif_df.loc[vif_df["VIF"].idxmax(), "Feature"]
        print(f"Removing {max_vif_feature} with VIF: {max_vif}")
        # stepwise
        data = data.drop(columns=[max_vif_feature])
        removed_features.append(max_vif_feature)
    reduced_df = pd.concat([data, df[drop_columns]], axis=1)
    return reduced_df, removed_features

def pearson_correlation_analysis(df, target_column, alpha=0.05):
    correlation_results = []
    removed_features = []
    for column in df.columns:
        if column != target_column:
            # calculate pearson
            corr, p_value = pearsonr(df[column], df[target_column])
            correlation_results.append({
                'Feature': column,
                'Correlation': corr,
                'P-value': p_value
            })
            if p_value >= alpha:
                removed_features.append(column)
    correlation_df = pd.DataFrame(correlation_results)
    print("Correlation Results:\n", correlation_df)
    # (p-value < alpha)
    significant_features = correlation_df[correlation_df['P-value'] < alpha]['Feature'].tolist()
    print(f"Significant Features (P-value < {alpha}): {significant_features}")
    print(f"Removed Features (P-value >= {alpha}): {removed_features}")
    filtered_df = df[significant_features + [target_column]]
    return filtered_df, correlation_df, removed_features

#random forest
def random_forest(X_train, y_train, X_test, y_test, importance_threshold=0.01):
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    selected_features = feature_importances[feature_importances['Importance'] > importance_threshold]['Feature']
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=1)
    grid_search.fit(X_train_selected, y_train)
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_selected)
    report = classification_report(y_test, y_pred)
    return best_model, report, selected_features
def SVM(X_train, y_train, X_test, y_test):
    svm = SVC(random_state=42, class_weight='balanced')
    param_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto']
    }
    grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=1)
    grid_search.fit(X_train, y_train)
    print("Best parameters found:")
    best_params = grid_search.best_params_
    print(best_params)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print("Classification Report:")
    report = classification_report(y_test, y_pred)
    print(report)
    return best_model, report
def gbc_classification(X_train, y_train, X_test, y_test):
    gbc = GradientBoostingClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    grid_search = GridSearchCV(estimator=gbc, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=1)
    grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    best_params = grid_search.best_params_
    print("Best Parameters Found:")
    print(best_params)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
    return best_model, report
def ann_classification(X_train, y_train, X_test, y_test):
    # 超参数网格
    param_grid = {
        'hidden_layer_sizes': [(50,),(100,), (100, 50), (50, 50)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [200, 500]
    }

    # 网格搜索
    ann = MLPClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=ann, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=1)
    grid_search.fit(X_train, y_train)

    # 最优模型和评估
    best_ann = grid_search.best_estimator_
    print("Best Parameters Found (ANN):", grid_search.best_params_)

    y_pred = best_ann.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("Classification Report (ANN):")
    print(report)

    return best_ann, report


if __name__ == "__main__":
    url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url_red, sep=";")
    class_counts = df['quality'].value_counts()
    print("各类别样本数：")
    print(class_counts)
    numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.difference(['quality'])
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    #Multicollinearity
    df, removed_features = remove_high_vif(df, drop_columns=['quality'], threshold=10)
    print("Removed Features(Multicollinearity):")
    print(removed_features)
    print(df.head())
    #pearson correlation analysis
    df, correlation_results, removed_features = pearson_correlation_analysis(df, target_column='quality', alpha=0.05)
    df['quality_label'] = pd.cut(
        df['quality'],
        bins=[3, 4, 6, 9],  
        labels=['low', 'medium', 'high'],
        include_lowest=True
    )
    class_counts = df['quality_label'].value_counts()
    print("各类别样本数：")
    print(class_counts)
    X = df.drop(columns=['quality', 'quality_label']) 
    y = df['quality_label'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    best_rf_model, rf_report, selected_features = random_forest(
        X_train, y_train, X_test, y_test, importance_threshold=0.01
    )
    print("Selected Features:")
    print(selected_features)
    print(rf_report)
    gbc_classification(X_train, y_train, X_test, y_test)
    gpc_model, gpc_report = gpc_classification(X_train, y_train, X_test, y_test)
    nb_model, nb_report = nb_classification(X_train, y_train, X_test, y_test)
    ann_model, ann_report = ann_classification(X_train, y_train, X_test, y_test)
