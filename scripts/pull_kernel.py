import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

print("Pulling kernel raykkretzschmar/kaggriculture-findings-from-zero-to-top-meta...")
api.kernels_pull("raykkretzschmar/kaggriculture-findings-from-zero-to-top-meta", path="D:\\DunderCode\\kaggle-kaggriculture\\submissions")
print("Done!")
