using Microsoft.AspNetCore.Mvc;

namespace HousingPrice.Api.Controllers;

[ApiController]
public class HealthController : ControllerBase
{
    [HttpGet("/health")]
    public IActionResult Get() => Ok(new { status = "healthy" });
}
