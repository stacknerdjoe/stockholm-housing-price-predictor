using System.Text.Json.Serialization;

namespace HousingPrice.Api.Services;

public sealed class ModelMetadata
{
    [JsonPropertyName("version")]
    public string Version { get; set; } = string.Empty;

    [JsonPropertyName("trained_at")]
    public string TrainedAt { get; set; } = string.Empty;

    [JsonPropertyName("area_values")]
    public List<string> AreaValues { get; set; } = [];

    [JsonPropertyName("onnx_filename")]
    public string OnnxFilename { get; set; } = string.Empty;

    [JsonPropertyName("onnx_opset")]
    public int OnnxOpset { get; set; }
}
