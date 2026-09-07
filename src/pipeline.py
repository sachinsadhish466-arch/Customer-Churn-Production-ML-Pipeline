from src.data.data_validator import validate_data
from src.data.data_cleaner import clean_data
from src.features.feature_engineering import create_features
from src.models.model_predictor import ChurnPredictor


class ChurnPipeline:

    def __init__(self, predictor=None):

        if predictor is not None:
            self.predictor = predictor
        else:
            self.predictor = ChurnPredictor()

    def prepare_data(self, df):

        validate_data(
            df,
            require_target=False
        )

        df = clean_data(df)

        df = create_features(df)

        return df

    def predict(self, df):

        df = self.prepare_data(df)

        X = df.copy()

        predictions, probabilities = (
            self.predictor.predict_with_probability(X)
        )

        return predictions, probabilities