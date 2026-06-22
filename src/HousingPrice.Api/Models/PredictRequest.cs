namespace HousingPrice.Api.Models;

public class PredictRequest
{
    public string Area { get; set; } = string.Empty;
    public int Rooms { get; set; }
    public double Size { get; set; }
    public decimal MonthlyFee { get; set; }
}
