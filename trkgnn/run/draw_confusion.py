import matplotlib
matplotlib.use('Agg')  # 使用Agg后端（无需GUI支持）
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 后续绘图代码不变
TP = 81866
FN = 201544834
FP = 9455885
TN = 865491

confusion_matrix = np.array([[TP, TN],
                             [FP, FN]])

plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Positive', 'Predicted Negative'], 
            yticklabels=['Actual Positive', 'Actual Negative'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()