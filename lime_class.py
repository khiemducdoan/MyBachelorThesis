import os
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import lime.lime_tabular
from sklearn.feature_selection import VarianceThreshold

import numpy as np
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


class LimeClassifier:
    def __init__(self):
        # Load model
        self.model = torch.load(os.path.join(current_dir, "naim_params.pt"), weights_only=False)
        self.model.eval()
        self.model = self.model.to('cuda')
        
        # Load and preprocess data
        data = pd.read_csv("/home/khanhhiep/Code/Khanh/Khiem/MyBachelorThesis/dataset/raw/datasetok.csv")
        self.X = data.drop(columns=["d_target", "text"])
        self.y = data["d_target"]
        self.feature_names = self.X.columns.tolist()
        self.class_names = ['rất nhẹ', 'nhẹ', 'trung bình', 'nặng']
        
        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        self.X_train = X_train.fillna(0)
        self.X_val = X_val.fillna(0)

        # Setup LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.feature_names,
            class_names=self.class_names,
            discretize_continuous=True
        )

    def predict_np(self, instance_numpy):
        """Predict class probabilities for input instance(s)."""
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.no_grad():
            input_tensor = torch.tensor(instance_numpy, dtype=torch.float32).to(device)
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            return probs.cpu().numpy()

    def explain(self, instance_numpy):
        """Generate a LIME explanation for a single input instance."""
        exp = self.explainer.explain_instance(
            instance_numpy,
            self.predict_np,
            num_features=len(self.feature_names),
            top_labels=1
        )
        label = int(list(exp.intercept.keys())[0])
        label_text = self.class_names[label]
        explanation_json = {
            "class_number": int(label),
            "class_text": label_text,
            "predicted_value": float(exp.predict_proba[label]),
            "intercept": float(exp.intercept[label]),
            "feature_weights": [
                {
                    "feature": self.feature_names[int(feat_idx)],
                    "weight": float(weight)
                }
                for feat_idx, weight in exp.local_exp[label]
            ],
            "explanation": [
                {
                    "description": desc,
                    "weight": float(w)
                }
                for desc, w in exp.as_list(label=label)
            ]
        }
        return explanation_json


        
    def predict(self, feature_vector):
        vector = []
        missed = []
        for i,fea in enumerate(self.feature_names):
            val = ''
            if fea in feature_vector:
                if feature_vector[fea] != '':
                    val = feature_vector[fea]
                else: 
                    missed.append(fea)
            else:
                missed.append({'fea':fea, 'id':i+1})
            val = str(val).lower()
            try:
                if 'có' in val:
                    val = 1
                elif 'không' in val:
                    val = 0
                else:
                    val = float(val)
            except:
                val = 0
            # print ('fea',fea, val)
            vector.append(val)
        
        vector_np = np.array(vector)
        # print ("vector_np",vector_np,vector_np.shape)
        vector_np = vector_np.reshape(1,-1)
        prediction = self.predict_np(vector_np)
        res_obj = self.explain(vector_np[0])
        res_obj['probability'] = prediction[0].tolist()
        res_obj['model'] = "LIME"
        return res_obj
        