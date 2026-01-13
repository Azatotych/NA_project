Start-Process -FilePath "uvicorn" -ArgumentList "backend.main:app --host 0.0.0.0 --port 8000" -PassThru | Out-Null
Set-Location frontend
npm install
npm run dev
