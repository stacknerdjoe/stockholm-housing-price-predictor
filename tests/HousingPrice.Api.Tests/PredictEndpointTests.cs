using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;

namespace HousingPrice.Api.Tests;

public class PredictEndpointTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;

    public PredictEndpointTests(WebApplicationFactory<Program> factory)
    {
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task PostPredict_WithValidInput_ReturnsEstimatedPriceInSEK()
    {
        // "Sodermalm" (no diacritic) is the spelling from the original API spec example.
        // This also exercises area normalisation: it must resolve to "Södermalm".
        var body = new { area = "Sodermalm", rooms = 2, size = 55, monthlyFee = 3200 };

        var response = await _client.PostAsJsonAsync("/predict", body);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var result = JsonSerializer.Deserialize<JsonElement>(
            await response.Content.ReadAsStringAsync());
        Assert.True(result.TryGetProperty("estimatedPrice", out var price));
        Assert.True(price.GetDecimal() > 0);
        Assert.Equal("SEK", result.GetProperty("currency").GetString());
    }

    [Fact]
    public async Task PostPredict_DiacriticVariant_ProducesSamePriceAsCanonicalSpelling()
    {
        // Verifies that "Sodermalm" and "Södermalm" reach the ONNX session with
        // the same canonical string and therefore return identical predictions.
        var shared = new { rooms = 3, size = 75, monthlyFee = 4100 };
        var withDiacritic    = new { area = "Södermalm", shared.rooms, shared.size, shared.monthlyFee };
        var withoutDiacritic = new { area = "Sodermalm", shared.rooms, shared.size, shared.monthlyFee };

        var r1 = await _client.PostAsJsonAsync("/predict", withDiacritic);
        var r2 = await _client.PostAsJsonAsync("/predict", withoutDiacritic);

        Assert.Equal(HttpStatusCode.OK, r1.StatusCode);
        Assert.Equal(HttpStatusCode.OK, r2.StatusCode);

        var p1 = JsonSerializer.Deserialize<JsonElement>(await r1.Content.ReadAsStringAsync())
            .GetProperty("estimatedPrice").GetDecimal();
        var p2 = JsonSerializer.Deserialize<JsonElement>(await r2.Content.ReadAsStringAsync())
            .GetProperty("estimatedPrice").GetDecimal();

        Assert.Equal(p1, p2);
    }

    [Fact]
    public async Task PostPredict_UnknownArea_Returns400WithErrorMessage()
    {
        var body = new { area = "Atlantis", rooms = 2, size = 55, monthlyFee = 3200 };

        var response = await _client.PostAsJsonAsync("/predict", body);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        var result = JsonSerializer.Deserialize<JsonElement>(
            await response.Content.ReadAsStringAsync());
        Assert.True(result.TryGetProperty("error", out var error));
        Assert.Contains("Atlantis", error.GetString());
    }
}
