"""
评估指标
"""
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, hamming_loss
)


class MultiLabelMetrics:
    """多标签分类指标"""
    
    def compute(self, y_true, y_pred, y_prob=None):
        """
        计算多标签分类指标
        """
        # 计算子集准确率 (所有标签都正确才算正确)
        subset_acc = np.mean(np.all(y_true == y_pred, axis=1))
        
        results = {
            'accuracy': subset_acc,  # 使用子集准确率
            'hamming_loss': hamming_loss(y_true, y_pred),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_samples': f1_score(y_true, y_pred, average='samples', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        }
        
        # 计算 AUC (如果提供概率)
        if y_prob is not None:
            try:
                results['auc_micro'] = roc_auc_score(
                    y_true, y_prob, average='micro', multi_class='ovr'
                )
            except:
                results['auc_micro'] = None
        
        return results
    
    def compute_per_class(self, y_true, y_pred, class_names=None):
        """计算每个类别的指标"""
        n_classes = y_true.shape[1]
        results = []
        
        for i in range(n_classes):
            if class_names is not None:
                name = class_names[i]
            else:
                name = f"class_{i}"
            
            results.append({
                'name': name,
                'precision': precision_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'recall': recall_score(y_true[:, i], y_pred[:, i], zero_division=0),
                'f1': f1_score(y_true[:, i], y_pred[:, i], zero_division=0),
            })
        
        return results


class MetricsTracker:
    """指标追踪器 - 用于记录训练过程"""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
        }
    
    def update(self, train_loss, val_loss, train_metrics, val_metrics):
        """更新指标"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_metrics'].append(train_metrics)
        self.history['val_metrics'].append(val_metrics)
    
    def get_best_epoch(self, metric='val_loss'):
        """获取最佳 epoch"""
        if metric == 'val_loss':
            return np.argmin(self.history['val_loss'])
        else:
            return np.argmax([m.get(metric, 0) for m in self.history['val_metrics']])
    
    def summary(self):
        """返回指标摘要"""
        best_epoch = self.get_best_epoch()
        return {
            'best_epoch': best_epoch,
            'best_val_loss': self.history['val_loss'][best_epoch],
            'best_val_metrics': self.history['val_metrics'][best_epoch],
        }