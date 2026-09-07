from src.data.data_validator import validate_data
from src.data.data_cleaner import clean_data
from src.features.feature_engineering import create_features
from src.models.model_predictor import ChurnPredictor


class ChurnPipeline:
    """
    End-to-end customer churn inference pipeline.
    """

    def __init__(self):
        self.predictor = ChurnPredictor()

    def prepare_data(self, df):
        """
        Validate, clean, and engineer features.
        """

        validate_data(
            df,
            require_target=False
        )

        df = clean_data(df)

        df = create_features(df)

        return df

    def predict(self, df):
        """
        Run the complete customer churn prediction pipeline.
        """

        df = self.prepare_data(df)

        # API inference data does not contain Churn.
        X = df.copy()

        predictions, probabilities = (
            self.predictor.predict_with_probability(X)
        )

        return predictions, probabilities