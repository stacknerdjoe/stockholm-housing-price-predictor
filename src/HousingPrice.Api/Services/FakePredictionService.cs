using HousingPrice.Api.Models;

namespace HousingPrice.Api.Services;

// TODO: Replace with real ONNX model inference once the model is trained and exported.
public class FakePredictionService : IPredictionService
{
    private const decimal PricePerSqm = 85_000m; // rough Stockholm average placeholder

    public decimal Predict(PredictRequest request) => (decimal)request.Size * PricePerSqm;
}
