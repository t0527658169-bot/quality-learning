# ===================================================
# Dockerfile — Audio Receiver (C# .NET 8)
# ===================================================
# שלב 1: Build
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# העתקת ה-csproj ושחזור חבילות (layer caching)
COPY AudioReceiver.csproj .
RUN dotnet restore

# העתקת כל שאר הקוד ובנייה
COPY . .
RUN dotnet publish -c Release -o /app/publish --no-restore

# ===================================================
# שלב 2: Runtime
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS final
WORKDIR /app

# תיקיה לקבצי השמע
RUN mkdir -p /app/audio /app/wwwroot

# העתקת הפרסום מהשלב הקודם
COPY --from=build /app/publish .

# הגדרת port
EXPOSE 5002
ENV ASPNETCORE_URLS=http://+:5002

ENTRYPOINT ["dotnet", "AudioReceiver.dll"]
