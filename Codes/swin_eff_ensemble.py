import pandas as pd

# 1. Load your Predictions
# (Make sure to upload both CSVs to your notebook environment)
df_eff = pd.read_csv("submission2.csv")         # The Robust Expert
df_swin = pd.read_csv("submission5.csv")   # The Clean Expert

# 2. Merge
# This ensures we are matching the exact same image to its score
df_ensemble = pd.merge(df_eff, df_swin, on="image_name")

# 3. Weighted Blending
# Since you noticed EffNet is better on robust (hard) images, we give it slightly more weight.
# Weight: 0.6 for EffNet (Reliable) / 0.4 for Swin (High Potential)
df_ensemble["final_score"] = (0.6 * df_eff["score"]) + (0.4 * df_swin["score"])

# 4. Save Final Submission
submission = df_ensemble[["image_name", "final_score"]]
submission.columns = ["image_name", "score"] # Rename for Kaggle
submission.to_csv("submission6.csv", index=False)

print("Ensemble Generated. Ready for the leaderboard!")