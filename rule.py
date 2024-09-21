import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import List, Tuple, Optional

class CounterpartyIdentifier:
    def __init__(self, input_file: str, reference_file: str, target_column: str = 'counterparty'):
        self.input_data = pd.read_csv(input_file)
        self.reference_data = pd.read_csv(reference_file)
        self.target_column = target_column
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoders = {}
        self.important_features = []

    def preprocess_data(self):
        # Combine input and reference data
        combined_data = pd.concat([self.input_data, self.reference_data], axis=0)
        
        # Encode categorical variables
        for column in combined_data.columns:
            if combined_data[column].dtype == 'object':
                le = LabelEncoder()
                combined_data[column] = le.fit_transform(combined_data[column].astype(str))
                self.label_encoders[column] = le

        # Split the data back into input and reference
        self.processed_input = combined_data.iloc[:len(self.input_data)]
        self.processed_reference = combined_data.iloc[len(self.input_data):]

    def train_model(self):
        X = self.processed_reference.drop(columns=[self.target_column])
        y = self.processed_reference[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)

        # Get feature importances
        importances = self.model.feature_importances_
        feature_importances = sorted(zip(importances, X.columns), reverse=True)
        self.important_features = [feature for _, feature in feature_importances]

    def identify_counterparty(self, row: pd.Series, max_attempts: int = 5) -> Tuple[Optional[str], List[str]]:
        for i in range(max_attempts):
            query_features = self.important_features[:len(self.important_features) - i]
            query = row[query_features].to_frame().T

            # Handle missing values
            query = query.fillna(query.mean())

            probabilities = self.model.predict_proba(query)[0]
            top_predictions = np.argsort(probabilities)[::-1][:2]  # Get top 2 predictions

            if probabilities[top_predictions[0]] > 0.5 and probabilities[top_predictions[0]] > 2 * probabilities[top_predictions[1]]:
                predicted_counterparty = self.label_encoders[self.target_column].inverse_transform([top_predictions[0]])[0]
                return predicted_counterparty, query_features

        return None, query_features

    def process_input_file(self) -> pd.DataFrame:
        results = []
        for _, row in self.processed_input.iterrows():
            counterparty, used_features = self.identify_counterparty(row)
            results.append({
                'identified_counterparty': counterparty,
                'used_features': ', '.join(used_features),
                'status': 'Success' if counterparty else 'Error: Multiple or no counterparties found'
            })
        return pd.DataFrame(results)

def main(input_file: str, reference_file: str) -> pd.DataFrame:
    identifier = CounterpartyIdentifier(input_file, reference_file)
    identifier.preprocess_data()
    identifier.train_model()
    return identifier.process_input_file()

# Usage
input_file = 'path/to/input.csv'
reference_file = 'path/to/reference.csv'
results = main(input_file, reference_file)
print(results)
