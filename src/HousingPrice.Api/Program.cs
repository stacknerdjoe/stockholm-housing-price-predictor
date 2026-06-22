using HousingPrice.Api.Services;
using Prometheus;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddControllers();

// Register as singleton so the InferenceSession is created once at startup.
// Both the concrete type (for ModelInfoController) and the interface
// (for PredictController) resolve to the same instance.
builder.Services.AddSingleton<OnnxPredictionService>();
builder.Services.AddSingleton<IPredictionService>(
    sp => sp.GetRequiredService<OnnxPredictionService>());

var app = builder.Build();

app.UseRouting();
app.UseHttpMetrics();
app.MapControllers();
app.MapMetrics();

app.Run();

// Exposes Program to the test project for WebApplicationFactory<Program>.
public partial class Program { }
