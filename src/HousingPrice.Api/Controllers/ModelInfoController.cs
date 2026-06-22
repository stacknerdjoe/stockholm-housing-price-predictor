using HousingPrice.Api.Services;
using Microsoft.AspNetCore.Mvc;

namespace HousingPrice.Api.Controllers;

[ApiController]
public class ModelInfoController : ControllerBase
{
    private readonly ModelMetadata _metadata;

    public ModelInfoController(OnnxPredictionService svc)
    {
        _metadata = svc.Metadata;
    }

    [HttpGet("/model-info")]
    public IActionResult Get() => Ok(new
    {
        modelLoaded  = true,
        version      = _metadata.Version,
        trainedAt    = _metadata.TrainedAt,
        validAreas   = _metadata.AreaValues,
    });
}
