// ===================================================
// Program.cs — נקודת כניסה לשירות C# (AudioReceiver)
// ===================================================
// קובץ זה הוא נקודת הכניסה של כל השרת.
//  הכל מוגדר כאן.
//
// סדר הרצה:
//   1. יצירת WebApplicationBuilder (הכנת כל השירותים)
//   2. הגדרת CORS, DB, Queue, HttpClient, Controllers, Swagger
//   3. בניית app = WebApplication.Build()
//   4. הגדרת pipeline של middleware (סדר חשוב!)
//   5. app.Run() — מפעיל השרת ומחכה לבקשות (infinite loop)
//
// כתובת: https://localhost:5002
// ===================================================

// ייבוא AppDbContext — חיבור EF Core ל-SQL Server
using AudioReceiver.Data;
// ייבוא שירותי התור ועובד הרקע
using AudioReceiver.Services;
// ייבוא Entity Framework Core ו-SQL Server Provider
using Microsoft.EntityFrameworkCore;

// WebApplication.CreateBuilder — יוצר builder עם ברירות מחדל של ASP.NET Core
// args = פרמטרי שורת הפקודה (מאפשר overrides ב-CLI)
var builder = WebApplication.CreateBuilder(args);


// (הפרונטאנד שולח בקשות ל-https://localhost:5002)
builder.WebHost.UseUrls("https://localhost:5002");

// -------------------------------------------------------
// CORS — Cross-Origin Resource Sharing
// -------------------------------------------------------
// CORS = מנגנון אבטחה של הדפדפן שמונע בקשות מ-origin שונה
// בלי הגדרת CORS, הפרונטאנד (שרץ ב-port 3000) לא יכול לשלוח
// בקשות לשרת ב-port 5002 — הדפדפן יחסום
// AllowAnyOrigin + AllowAnyHeader + AllowAnyMethod = פתוח לכל
// (מקובל ב-development; בפרודקשן כדאי להגביל ל-domain ספציפי)
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
        policy.AllowAnyOrigin()
              .AllowAnyHeader()
              .AllowAnyMethod());
});

// -------------------------------------------------------
// חיבור ל-SQL Server דרך EF Core
// -------------------------------------------------------
// GetConnectionString("DefaultConnection") = קורא מ-appsettings.json:
//   "ConnectionStrings": { "DefaultConnection": "Server=...;Database=..." }
// אם לא נמצא — בונה connection string ממשתני סביבה (לסביבת Docker/Production)
var connectionString = builder.Configuration.GetConnectionString("DefaultConnection")
    ?? BuildConnectionStringFromEnv();

// AddDbContext<AppDbContext> = רישום AppDbContext כשירות Scoped
// Scoped = instance חדש לכל בקשת HTTP (thread-safe)
// UseSqlServer = שימוש ב-SQL Server Provider
// EnableRetryOnFailure(3) = ניסיון חוזר עד 3 פעמים אם DB לא זמין זמנית
builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseSqlServer(connectionString,
        sqlOptions => sqlOptions.EnableRetryOnFailure(3)));

// -------------------------------------------------------
// שירותי האפליקציה — Dependency Injection (DI)
// -------------------------------------------------------

// AddSingleton = instance אחד לכל חיי האפליקציה (משותף לכל threads)
// נדרש כי ConcurrentQueue חייב להיות משותף בין:
//   RecordingsController (מוסיף להקלטות לתור)
//   QueueProcessorService (שולף הקלטות לעיבוד)
builder.Services.AddSingleton<IAudioQueueService, AudioQueueService>();

// AddHostedService = רישום BackgroundService שרץ בthread נפרד
// ASP.NET Core יפעיל QueueProcessorService.ExecuteAsync() אוטומטית בסטארטאפ
builder.Services.AddHostedService<QueueProcessorService>();

// AddHttpClient("python") = רישום HttpClient מוגדר לתקשורת עם Python
// IHttpClientFactory מנהל connection pooling ומונע socket exhaustion
builder.Services.AddHttpClient("python", client =>
{
    // קריאת כתובת Python מ-appsettings.json או ברירת מחדל
    // בפיתוח מקומי: http://localhost:5000
    // בDocker: http://audio-processor:5000 (שם השירות ב-docker-compose)
    var pythonUrl = builder.Configuration["AudioProcessor:Url"] ?? "http://audio-processor:5000";
    // חובה: trailing slash כדי ש-relative paths יעבדו נכון ב-.NET HttpClient
    // ללא ה-/ הסיומת: new Uri("http://host:5000") + "process" = http://host:process (שגוי!)
    // עם ה-/ הסיומת: new Uri("http://host:5000/") + "process" = http://host:5000/process (נכון!)
    if (!pythonUrl.EndsWith("/")) pythonUrl += "/";
    client.BaseAddress = new Uri(pythonUrl);
    // Timeout ארוך — Python עשוי לקחת עד 2 שעות לעיבוד הקלטה ארוכה
    client.Timeout = TimeSpan.FromMinutes(120);
});

// AddControllers = רישום כל ה-Controllers (Auth + Recordings) כשירותים
builder.Services.AddControllers();
// AddEndpointsApiExplorer = נדרש ל-Swagger (בודק את ה-endpoints)
builder.Services.AddEndpointsApiExplorer();
// AddSwaggerGen = יצירת תיעוד OpenAPI אינטראקטיבי
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new()
    {
        Title = "Quality Learning — Audio Receiver API",
        Description = "מקבל הקלטות שמע, מכניס לתור, ושולח לניתוח פייתון"
    });
});

// Build() — בונה את האפליקציה מכל ההגדרות שנקבעו
// לאחר Build() לא ניתן להוסיף שירותים חדשים
var app = builder.Build();

// -------------------------------------------------------
// בדיקת חיבור ל-DB בהפעלה
// -------------------------------------------------------
// CreateScope = scope חדש לשימוש חד-פעמי (AppDbContext הוא Scoped)
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
    // CanConnectAsync() — שולח בקשת TCP+SQL פשוטה לשרת הנתונים
    // אם ה-DB לא זמין → זורק Exception → השרת לא עולה (כוונתי!)
    // מאפשר לאתר בעיות חיבור מיד בסטארטאפ ולא בזמן בקשה ראשונה
    // שים לב: הטבלאות נוצרות ע"י סקריפט SQL ב-SSMS (לא EF Migrations)
    await db.Database.CanConnectAsync();
}

// -------------------------------------------------------
// Middleware Pipeline — סדר ביצוע חשוב!
// כל בקשת HTTP עוברת דרך כל שכבה בסדר הזה
// -------------------------------------------------------

// UseCors — חובה לפני MapControllers!
// מוסיף headers של CORS לכל תשובה (Access-Control-Allow-Origin וכו')
app.UseCors();

// UseSwagger — מוסיף endpoint /swagger/v1/swagger.json (JSON של ה-API)
app.UseSwagger();
// UseSwaggerUI — מוסיף ממשק Swagger אינטראקטיבי ב-/swagger
// מאפשר לבדוק כל endpoint ישירות מהדפדפן ללא Postman
app.UseSwaggerUI();

// UseDefaultFiles — מפנה בקשות ל-/ → /index.html (אוטומטי)
// חובה לפני UseStaticFiles!
app.UseDefaultFiles();
// UseStaticFiles — מגיש קבצים סטטיים מתיקיית wwwroot:
// JS, CSS, תמונות, index.html — הפרונטאנד כולו
app.UseStaticFiles();

// MapControllers — מחבר נתיבי URL לפונקציות ב-Controllers
// RecordingsController: /api/recordings/...
// AuthController: /api/auth/...
app.MapControllers();

// MapFallbackToFile — "לכידת" כל נתיב שלא נמצא ב-API ל-index.html
// הכרחי ל-SPA (Single Page Application): כשהמשתמש מרענן את הדפדפן
// ב-URL כמו /dashboard, השרת צריך להחזיר index.html (לא 404)
// React/Vue מטפל בניתוב בצד הלקוח
app.MapFallbackToFile("index.html");

// app.Run() — מפעיל את השרת ומחכה לבקשות HTTP
// זה חוסם את ה-thread הראשי עד לעצירה (Ctrl+C)
// בינתיים: QueueProcessorService רץ ב-thread נפרד ברקע
app.Run();

// -------------------------------------------------------
// פונקציה עזר — בונה connection string ל-SQL Server ממשתני סביבה
// -------------------------------------------------------
static string BuildConnectionStringFromEnv()
{
    // משתני סביבה מאפשרים הגדרת DB שונה בפרודקשן בלי לשנות קוד
    var server = Environment.GetEnvironmentVariable("DB_SERVER") ?? "localhost";
    var dbName = Environment.GetEnvironmentVariable("DB_NAME") ?? "QualityLearning";
    var user = Environment.GetEnvironmentVariable("DB_USER") ?? "";
    var password = Environment.GetEnvironmentVariable("DB_PASSWORD") ?? "";

    // אם יש שם משתמש — SQL Auth (username+password), אחרת Windows Auth (Trusted_Connection)
    if (!string.IsNullOrEmpty(user))
        return $"Server={server};Database={dbName};User Id={user};Password={password};TrustServerCertificate=True;";
    else
        return $"Server={server};Database={dbName};Trusted_Connection=True;TrustServerCertificate=True;";
}
