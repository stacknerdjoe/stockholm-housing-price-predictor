# ── Build stage ────────────────────────────────────────────────────────────────
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# Copy only the project file first so the restore layer is cached as long as
# dependencies don't change (source edits don't bust this layer).
COPY src/HousingPrice.Api/HousingPrice.Api.csproj src/HousingPrice.Api/
RUN dotnet restore src/HousingPrice.Api/HousingPrice.Api.csproj \
    --runtime linux-x64

COPY src/ src/
RUN dotnet publish src/HousingPrice.Api/HousingPrice.Api.csproj \
    --configuration Release \
    --runtime linux-x64 \
    --no-self-contained \
    --output /app/publish \
    --no-restore

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS runtime
WORKDIR /app

# Published .NET output
COPY --from=build /app/publish .

# ONNX model and metadata.
# OnnxPredictionService.FindModelsDirectory walks up from ContentRootPath (/app),
# so placing models/ here means it is found on the first check.
COPY models/ models/

ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080

ENTRYPOINT ["dotnet", "HousingPrice.Api.dll"]
