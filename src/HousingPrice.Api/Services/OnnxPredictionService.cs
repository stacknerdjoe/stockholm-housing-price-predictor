using System.Globalization;
using System.Text;
using System.Text.Json;
using HousingPrice.Api.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace HousingPrice.Api.Services;

// Loads the ONNX model once at startup and serves all predict requests.
// Registered as singleton so InferenceSession is shared across requests.
public sealed class OnnxPredictionService : IPredictionService, IDisposable
{
    private readonly InferenceSession _session;
    private readonly ILogger<OnnxPredictionService> _logger;

    // Maps diacritic-stripped lowercase area -> exact training spelling.
    // Built once from metadata; used to normalise incoming API strings.
    private readonly IReadOnlyDictionary<string, string> _areaMap;

    public ModelMetadata Metadata { get; }

    public OnnxPredictionService(
        IWebHostEnvironment env,
        ILogger<OnnxPredictionService> logger)
    {
        _logger = logger;

        var modelsDir = FindModelsDirectory(env.ContentRootPath);

        var metaPath = Path.Combine(modelsDir, "model-metadata.json");
        Metadata = JsonSerializer.Deserialize<ModelMetadata>(
            File.ReadAllText(metaPath),
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
            ?? throw new InvalidOperationException(
                $"Failed to deserialize {metaPath}");

        var onnxPath = Path.Combine(modelsDir, Metadata.OnnxFilename);
        _session = new InferenceSession(onnxPath);

        _areaMap = Metadata.AreaValues.ToDictionary(
            area => StripDiacritics(area),
            area => area,
            StringComparer.OrdinalIgnoreCase);

        _logger.LogInformation(
            "ONNX model loaded. Version={Version}, Opset={Opset}, Areas={AreaCount}",
            Metadata.Version, Metadata.OnnxOpset, Metadata.AreaValues.Count);
    }

    public decimal Predict(PredictRequest request)
    {
        // Normalise: case-insensitive + diacritic-insensitive match against trained areas.
        var key = StripDiacritics(request.Area);
        if (!_areaMap.TryGetValue(key, out var canonicalArea))
        {
            var valid = string.Join(", ", Metadata.AreaValues);
            throw new ArgumentException(
                $"Unknown area '{request.Area}'. Valid areas: {valid}",
                nameof(request));
        }

        // Input shapes match export_onnx.py's initial_types: each column is [1, 1].
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("area",
                new DenseTensor<string>(
                    new Memory<string>(new[] { canonicalArea }),
                    new[] { 1, 1 })),

            NamedOnnxValue.CreateFromTensor("rooms",
                new DenseTensor<long>(
                    new Memory<long>(new[] { (long)request.Rooms }),
                    new[] { 1, 1 })),

            NamedOnnxValue.CreateFromTensor("size",
                new DenseTensor<double>(
                    new Memory<double>(new[] { request.Size }),
                    new[] { 1, 1 })),

            NamedOnnxValue.CreateFromTensor("monthlyFee",
                new DenseTensor<long>(
                    new Memory<long>(new[] { (long)request.MonthlyFee }),
                    new[] { 1, 1 })),
        };

        using var results = _session.Run(inputs);

        // Output node is named 'variable', shape [1, 1], dtype float32.
        var output = results.First(r => r.Name == "variable").AsTensor<float>();
        return (decimal)output[0, 0];
    }

    public void Dispose() => _session.Dispose();

    // Decompose to NFD then drop combining marks (NonSpacingMark category).
    // "Södermalm" and "Sodermalm" both reduce to "sodermalm".
    private static string StripDiacritics(string text)
    {
        var nfd = text.Normalize(NormalizationForm.FormD);
        return new string(
            nfd.Where(c => CharUnicodeInfo.GetUnicodeCategory(c)
                           != UnicodeCategory.NonSpacingMark)
               .ToArray());
    }

    // Walk up the directory tree from startPath until a models/ directory is
    // found that contains the ONNX file. Works from any working directory:
    // API content root in development, test output dir during testing.
    private static string FindModelsDirectory(string startPath)
    {
        var dir = new DirectoryInfo(startPath);
        while (dir is not null)
        {
            var candidate = Path.Combine(dir.FullName, "models");
            if (Directory.Exists(candidate) &&
                File.Exists(Path.Combine(candidate, "housing-price-model.onnx")))
                return candidate;
            dir = dir.Parent;
        }
        throw new DirectoryNotFoundException(
            "Cannot find models/ directory containing housing-price-model.onnx. " +
            $"Searched from: {startPath}");
    }
}
