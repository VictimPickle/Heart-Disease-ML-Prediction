import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('heart.csv')
print(df.dtypes)
print(df.value_counts())

chol_missing = (df['Cholesterol'] == 0).sum()
restbp_missing = (df['RestingBP'] == 0).sum()
print("Missing Cholesterol (coded as 0):", chol_missing)
print("Missing RestingBP (coded as 0):", restbp_missing)

df_clean = df[(df['Cholesterol'] != 0) & (df['RestingBP'] != 0)].copy()

encoding_maps = {
    'Sex': {'M': 1, 'F': 0},
    'ChestPainType': {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3},
    'RestingECG': {'Normal': 0, 'ST': 1, 'LVH': 2},
    'ExerciseAngina': {'N': 0, 'Y': 1},
    'ST_Slope': {'Down': 0, 'Flat': 1, 'Up': 2}
}

df_encoded = df_clean.copy()
for feature, mapping in encoding_maps.items():
    df_encoded[feature] = df_encoded[feature].map(mapping)

corr_matrix = df_encoded.corr()
target_corr = corr_matrix['HeartDisease'].drop('HeartDisease').sort_values(key=abs, ascending=False)

X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

dt_raw = DecisionTreeClassifier(random_state=42)
dt_raw.fit(X_train, y_train)

y_train_pred_raw = dt_raw.predict(X_train)
y_test_pred_raw = dt_raw.predict(X_test)

raw_train_acc = accuracy_score(y_train, y_train_pred_raw)
raw_test_acc = accuracy_score(y_test, y_test_pred_raw)

print("Raw Decision Tree:")
print(f"  Train Accuracy: {raw_train_acc:.4f}")
print(f"  Test  Accuracy: {raw_test_acc:.4f}")

dt_model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    criterion='gini',
    random_state=42
)
dt_model.fit(X_train, y_train)

y_train_pred_dt = dt_model.predict(X_train)
y_test_pred_dt = dt_model.predict(X_test)

dt_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

y_train_pred_nb = nb_model.predict(X_train_scaled)
y_test_pred_nb = nb_model.predict(X_test_scaled)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_test_pred_rf = rf_model.predict(X_test)

rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
xgb_model.fit(X_train, y_train)
y_test_pred_xgb = xgb_model.predict(X_test)

def evaluate_model(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp
    }

dt_train_acc = accuracy_score(y_train, y_train_pred_dt)
dt_test = evaluate_model(y_test, y_test_pred_dt)

nb_train_acc = accuracy_score(y_train, y_train_pred_nb)
nb_test = evaluate_model(y_test, y_test_pred_nb)

rf_test = evaluate_model(y_test, y_test_pred_rf)
xgb_test = evaluate_model(y_test, y_test_pred_xgb)

print("MODEL EVALUATION - TEST SET")

print("\nDecision Tree:")
print(f"  Train Acc: {dt_train_acc:.4f} | Test Acc: {dt_test['Accuracy']:.4f}")
print(f"  Precision: {dt_test['Precision']:.4f} | Recall: {dt_test['Recall']:.4f} | F1: {dt_test['F1']:.4f}")
print(f"  CM: TN={dt_test['TN']}, FP={dt_test['FP']}, FN={dt_test['FN']}, TP={dt_test['TP']}")

print("\nNaive Bayes:")
print(f"  Train Acc: {nb_train_acc:.4f} | Test Acc: {nb_test['Accuracy']:.4f}")
print(f"  Precision: {nb_test['Precision']:.4f} | Recall: {nb_test['Recall']:.4f} | F1: {nb_test['F1']:.4f}")
print(f"  CM: TN={nb_test['TN']}, FP={nb_test['FP']}, FN={nb_test['FN']}, TP={nb_test['TP']}")

print("\nRandom Forest:")
print(f"  Precision: {rf_test['Precision']:.4f} | Recall: {rf_test['Recall']:.4f} | F1: {rf_test['F1']:.4f}")
print(f"  Accuracy: {rf_test['Accuracy']:.4f}")

print("\nXGBoost:")
print(f"  Precision: {xgb_test['Precision']:.4f} | Recall: {xgb_test['Recall']:.4f} | F1: {xgb_test['F1']:.4f}")
print(f"  Accuracy: {xgb_test['Accuracy']:.4f}")

comparison = pd.DataFrame({
    'Model': ['Decision Tree', 'Naive Bayes', 'Random Forest', 'XGBoost'],
    'Accuracy': [dt_test['Accuracy'], nb_test['Accuracy'], rf_test['Accuracy'], xgb_test['Accuracy']],
    'Precision': [dt_test['Precision'], nb_test['Precision'], rf_test['Precision'], xgb_test['Precision']],
    'Recall': [dt_test['Recall'], nb_test['Recall'], rf_test['Recall'], xgb_test['Recall']],
    'F1': [dt_test['F1'], nb_test['F1'], rf_test['F1'], xgb_test['F1']],
    'False_Negatives': [dt_test['FN'], nb_test['FN'], rf_test['FN'], xgb_test['FN']]
})

print("FINAL COMPARISON")
print(comparison.to_string(index=False))

best_idx = comparison['Recall'].idxmax()
print(f"\nBest model: {comparison.loc[best_idx, 'Model']}")

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('01_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

top_dt = dt_importance.head(10)
ax1.barh(range(len(top_dt)), top_dt['Importance'], color='steelblue')
ax1.set_yticks(range(len(top_dt)))
ax1.set_yticklabels(top_dt['Feature'])
ax1.set_title('Decision Tree - Top 10 Features')
ax1.invert_yaxis()

top_rf = rf_importance.head(10)
ax2.barh(range(len(top_rf)), top_rf['Importance'], color='coral')
ax2.set_yticks(range(len(top_rf)))
ax2.set_yticklabels(top_rf['Feature'])
ax2.set_title('Random Forest - Top 10 Features')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('02_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

models = ['DT', 'NB', 'RF', 'XGB']
colors = ['steelblue', 'coral', 'lightgreen', 'orange']

ax1.bar(models, comparison['Accuracy'], color=colors)
ax1.set_title('Accuracy')
ax1.set_ylim([0.7, 1])

ax2.bar(models, comparison['Precision'], color=colors)
ax2.set_title('Precision')
ax2.set_ylim([0.7, 1])

ax3.bar(models, comparison['Recall'], color=colors)
ax3.set_title('Recall')
ax3.set_ylim([0.7, 1])

ax4.bar(models, comparison['F1'], color=colors)
ax4.set_title('F1-Score')
ax4.set_ylim([0.7, 1])

plt.tight_layout()
plt.savefig('03_model_metrics.png', dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

cm_dt = confusion_matrix(y_test, y_test_pred_dt)
cm_nb = confusion_matrix(y_test, y_test_pred_nb)
cm_rf = confusion_matrix(y_test, y_test_pred_rf)
cm_xgb = confusion_matrix(y_test, y_test_pred_xgb)

sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False)
axes[0, 0].set_title('Decision Tree')

sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Oranges', ax=axes[0, 1], cbar=False)
axes[0, 1].set_title('Naive Bayes')

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1, 0], cbar=False)
axes[1, 0].set_title('Random Forest')

sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Purples', ax=axes[1, 1], cbar=False)
axes[1, 1].set_title('XGBoost')

plt.tight_layout()
plt.savefig('04_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, comparison['False_Negatives'], color=colors)
ax.set_title('False Negatives (Missed Disease Cases)')
ax.set_ylabel('Count')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('05_false_negatives.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(24, 14))
plot_tree(dt_model, feature_names=list(X.columns), class_names=['Healthy', 'Disease'],
          filled=True, fontsize=10, rounded=True, max_depth=3)
plt.title('Decision Tree (First 3 Levels)')
plt.tight_layout()
plt.savefig('06_decision_tree.png', dpi=150, bbox_inches='tight')
plt.show()

comparison.to_csv('model_comparison_results.csv', index=False)
dt_importance.to_csv('decision_tree_importance.csv', index=False)
rf_importance.to_csv('random_forest_importance.csv', index=False)
df_encoded.to_csv('processed_heart_data.csv', index=False)
