# preprocessor belum tentu sudah load() — pastikan artifacts dimuat
from app.preprocessing import preprocessor
import joblib, json

# Load preprocessing artifacts (uses artifacts dir from settings)
preprocessor.load()
print("Has medians:", bool(preprocessor.medians))
# tampilkan beberapa medians
for c in preprocessor.cols[:12]:
    print("median:", c, preprocessor.medians.get(c))

# tampilkan scaler center_ / scale_
scaler = preprocessor.scaler
print("scaler attr center_ exists:", hasattr(scaler, 'center_'))
if hasattr(scaler, 'center_'):
    print("scaler.center_ sample:", scaler.center_[:12])
if hasattr(scaler, 'scale_'):
    print("scaler.scale_ sample:", scaler.scale_[:12])