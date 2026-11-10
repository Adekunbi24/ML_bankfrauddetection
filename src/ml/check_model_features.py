import joblib

MODEL_PATH = "models/random_forest_model.joblib"

model = joblib.load(MODEL_PATH)

preprocessor = model.named_steps["preprocessing"]
classifier = model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()
importance_scores = classifier.feature_importances_

results = sorted(
    zip(feature_names, importance_scores),
    key=lambda item: item[1],
    reverse=True,
)

print("\nRandom Forest Feature Importance:")

for feature, score in results:
    print(f"{feature}: {score:.4f}")