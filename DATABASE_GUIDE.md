```powershell
@"
# PostgreSQL Baseline Workflow — Quick Reference

## First Time Setup (Once Per Project)

### 1. Verify PostgreSQL is Running
```powershell
Get-Service -Name "postgresql*"
```
Expected: Status = Running

### 2. Connect & Create Database
```powershell
`$env:PGPASSWORD = "your_password"
& "D:\PostgreSQL\18\data\bin\psql.exe" -U postgres -p 5433 -c "CREATE DATABASE your_project_db;"
```

### 3. Run Schema File
```powershell
& "D:\PostgreSQL\18\data\bin\psql.exe" -U postgres -p 5433 -d your_project_db -f "src/database/schema.sql"
```


## Daily Workflow (Every Session)

### 1. Start PostgreSQL (if not running)
```powershell
Start-Service -Name "postgresql-x64-18"
```

### 2. Verify Connection
```powershell
python -c "from src.database.connection import test_connection; test_connection()"
```
Expected: ✅ Database connection successful!

### 3. Check What's in the Database
```powershell
# Count rows in main tables
`$env:PGPASSWORD = "your_password"
& "D:\PostgreSQL\18\data\bin\psql.exe" -U postgres -p 5433 -d your_project_db -c "
SELECT 
    (SELECT COUNT(*) FROM transactions) AS transactions,
    (SELECT COUNT(*) FROM predictions) AS predictions,
    (SELECT COUNT(*) FROM ground_truth) AS ground_truth;"
```

### 4. Develop / Run Your Pipeline
```powershell
# Train:  python -m src.models.train_baseline
# Serve:  python -c "from src.serve import get_app; app = get_app(); app.run()"
# Eval:   python -m src.evaluation.evaluate --model models/best_model.pkl
# Monitor: python -m src.monitor
```

### 5. Verify New Data Was Saved
```powershell
& "D:\PostgreSQL\18\data\bin\psql.exe" -U postgres -p 5433 -d your_project_db -c "
SELECT table_name, 
       (SELECT COUNT(*) FROM transactions) AS row_count
FROM information_schema.tables 
WHERE table_schema = 'public';"
```


## Quick Debugging

### Connection Issues
```powershell
# Is PostgreSQL running?
Get-Service -Name "postgresql*"

# Is port open?
Test-NetConnection -ComputerName localhost -Port 5433

# Restart PostgreSQL
Restart-Service -Name "postgresql-x64-18"
```

### Check Recent Errors
```powershell
python -c "
from src.database.connection import engine
from sqlalchemy import text
with engine.connect() as conn:
    result = conn.execute(text('SELECT * FROM monitoring_metrics WHERE is_warning = true ORDER BY check_timestamp DESC LIMIT 5'))
    for row in result:
        print(row)
"
```

### View Latest Data
```powershell
& "D:\PostgreSQL\18\data\bin\psql.exe" -U postgres -p 5433 -d your_project_db -c "
SELECT * FROM vw_transaction_results ORDER BY timestamp_received DESC LIMIT 5;"
```


## Remember

| Rule | Why |
|------|-----|
| Always verify connection first | Saves debugging time |
| Check row counts after pipeline runs | Confirms data was saved |
| Use views for analysis, not raw tables | Views handle joins automatically |
| Keep schema.sql updated | Single source of truth for DB structure |
| Password in env vars, never in code | Security habit from day one |
"@ | Out-File -FilePath "POSTGRESQL_WORKFLOW.md" -Encoding ASCII
```

---

## File Created: `POSTGRESQL_WORKFLOW.md`

This file is now in your project root. You can reference it every time you start working.

Your daily habit becomes:
```powershell
# 1. Check PostgreSQL
Get-Service -Name "postgresql*"

# 2. Verify connection
python -c "from src.database.connection import test_connection; test_connection()"

# 3. Go!