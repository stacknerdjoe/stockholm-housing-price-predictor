namespace HousingPrice.Api.Models;

public class PredictResponse
{
    public decimal EstimatedPrice { get; set; }
    public string Currency { get; set; } = "SEK";
}
