import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Прикладні справжні та передбачені значення
# Дані (реальні та передбачені значення)
y_true = [0]*60 + [1]*420  # 60 реальних, 420 фейкових
y_pred = [0]*25 + [1]*35 + [0]*79 + [1]*341  # Згідно з матрицею

labels = ['REAL', 'FAKE']
cm = confusion_matrix(y_true, y_pred)
accuracy = (cm[0][0] + cm[1][1]) / np.sum(cm)

# Створення фігури з двома підграфіками: матриця + текстовий звіт
fig, ax = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1]})

# Матриця плутанини
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax[0])
ax[0].set_xlabel('Predicted label')
ax[0].set_ylabel('True label')
ax[0].set_title(f'Confusion Matrix for Facenet\n\nFacenet Accuracy: {accuracy:.15f}')

# Текстовий звіт
report = classification_report(y_true, y_pred, target_names=labels, digits=2)
ax[1].axis('off')  # Вимикаємо вісь
ax[1].text(0.0, 1.0, report, fontsize=12, family='monospace', va='top')

plt.tight_layout()
plt.show()