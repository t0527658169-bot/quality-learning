// ===================================================
// AuthController.cs — כניסה למערכת
// ===================================================
// מטפל בבקשת POST /api/auth/login.
// בודק שם משתמש + סיסמה מול טבלת Teachers ב-SQL Server.
// מחזיר { id, fullName, role } אם הפרטים נכונים.
//
// אבטחה:
//   - סיסמאות מאוחסנות כ-SHA256 hash (hex lowercase)
//   - פרמטרים מוגנים ב-parameterized query — מונע SQL Injection
//   - is_active = 1 מבטיח שמשתמש מושבת לא יוכל להתחבר
// ===================================================

// גישה למסד הנתונים דרך EF Core (AppDbContext)
using AudioReceiver.Data;
// מספק את ControllerBase ו-IActionResult
using Microsoft.AspNetCore.Mvc;
// Entity Framework Core — דרוש ל-GetDbConnection()
using Microsoft.EntityFrameworkCore;
// הצפנת SHA256 — לחישוב hash של הסיסמה
using System.Security.Cryptography;
// קידוד UTF8 לצורך ה-hash
using System.Text;

namespace AudioReceiver.Controllers;

// [ApiController] — מאפשר ולידציה אוטומטית ו-JSON responses
[ApiController]
// [Route] — הנתיב הוא api/auth (שם הבקר)
[Route("api/[controller]")]
public class AuthController : ControllerBase
{
    // הזרקת AppDbContext — חיבור לבסיס הנתונים, מסופק על ידי DI
    private readonly AppDbContext _db;
    // Constructor Injection — ASP.NET Core מזריק את _db אוטומטית
    public AuthController(AppDbContext db) => _db = db;

    // POST /api/auth/login
    // מקבל שם משתמש וסיסמה כ-Form fields (לא JSON, כי HTML form)
    [HttpPost("login")]
    public async Task<IActionResult> Login([FromForm] string username, [FromForm] string password)
    {
        // ולידציה בסיסית — שדות חובה לא ריקים
        // IsNullOrWhiteSpace בודק null, מחרוזת ריקה, ורווחים בלבד
        if (string.IsNullOrWhiteSpace(username) || string.IsNullOrWhiteSpace(password))
            return BadRequest(new { error = "שם משתמש וסיסמה הם שדות חובה" });

        // חישוב SHA256 hash של הסיסמה:
        // Encoding.UTF8.GetBytes → המרת string ל-bytes
        // SHA256.HashData → חישוב ה-hash (32 bytes)
        // Convert.ToHexString → המרה לייצוג hex (64 תווים)
        // .ToLower() → לאחידות עם הנתונים במסד (שמורים ב-lowercase)
        var hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(password))).ToLower();

        // קבלת חיבור SQL גולמי מ-EF Core (מהיר יותר מ-LINQ לשאילתה פשוטה)
        var conn = _db.Database.GetDbConnection();
        await conn.OpenAsync();
        try
        {
            using var cmd = conn.CreateCommand();
            // שאילתה עם 3 תנאים:
            //   1. username = @u — שם המשתמש (ת"ז / מזהה)
            //   2. password_hash = @h (SHA256) OR password_hash = @plain (לתמיכה בסיסמאות ישנות)
            //   3. is_active = 1 — משתמש פעיל בלבד
            // @u, @h, @plain הם placeholders — מוחלפים בבטחה ע"י ADO.NET (מונע SQL Injection)
            cmd.CommandText = "SELECT id, full_name, role FROM Teachers WHERE username = @u AND (password_hash = @h OR password_hash = @plain) AND is_active = 1";

            // יצירת פרמטר @u — שם המשתמש (לאחר Trim להסרת רווחים)
            var pu = cmd.CreateParameter(); pu.ParameterName = "@u";     pu.Value = username.Trim(); cmd.Parameters.Add(pu);
            // יצירת פרמטר @h — hash SHA256 של הסיסמה
            var ph = cmd.CreateParameter(); ph.ParameterName = "@h";     ph.Value = hash;            cmd.Parameters.Add(ph);
            // יצירת פרמטר @plain — הסיסמה הגולמית (fallback למשתמשים ישנים)
            var pp = cmd.CreateParameter(); pp.ParameterName = "@plain"; pp.Value = password;        cmd.Parameters.Add(pp);

            // הרצת השאילתה — ExecuteReaderAsync מחזיר שורות תוצאה
            using var reader = await cmd.ExecuteReaderAsync();
            // אם אין שורה — שם משתמש או סיסמה שגויים
            // 401 Unauthorized = לא מאומת
            if (!await reader.ReadAsync())
                return Unauthorized(new { error = "שם משתמש או סיסמה שגויים" });

            // נמצאה שורה — מחזירים את נתוני המשתמש:
            // GetInt32(0) = עמודה 0 = id
            // GetString(1) = עמודה 1 = full_name
            // GetString(2) = עמודה 2 = role (teacher / principal)
            return Ok(new { id = reader.GetInt32(0), fullName = reader.GetString(1), role = reader.GetString(2) });
        }
        finally
        {
            // סגירת החיבור תמיד — גם אם נזרקה חריגה
            // finally מבטיח ביצוע בכל מקרה
            await conn.CloseAsync();
        }
    }
}
