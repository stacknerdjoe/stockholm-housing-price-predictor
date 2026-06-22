using HousingPrice.Api.Models;
using HousingPrice.Api.Services;
using Microsoft.AspNetCore.Mvc;

namespace HousingPrice.Api.Controllers;

[ApiController]
public class PredictController : ControllerBase
{
    private readonly IPredictionService _predictionService;

    public PredictController(IPredictionService predictionService)
    {
        _predictionService = predictionService;
    }

    [HttpPost("/predict")]
    public IActionResult Post([FromBody] PredictRequest request)
    {
        try
        {
            var price = _predictionService.Predict(request);
            return Ok(new PredictResponse { EstimatedPrice = price, Currency = "SEK" });
        }
        catch (ArgumentException ex)
        {
            return BadRequest(new { error = ex.Message });
        }
    }
}
