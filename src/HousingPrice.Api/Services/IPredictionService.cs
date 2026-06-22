using HousingPrice.Api.Models;

namespace HousingPrice.Api.Services;

public interface IPredictionService
{
    decimal Predict(PredictRequest request);
}
