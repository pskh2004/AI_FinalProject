# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# نصب ابزارهای مورد نیاز سیستم‌عامل لینوکس
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# کپی فایل پیش‌نیازها به تنهایی جهت فعال‌سازی کش داکر
COPY requirements.txt .

# نصب لایبرری‌های پایتون
RUN pip install --no-cache-dir -r requirements.txt

# کپی کردن کدهای منبع، وب‌سرویس و وزن‌های مدل‌های آموزش‌دیده
COPY src/ ./src/
COPY serve_model.py .
COPY mlruns/ ./mlruns/

# توجه: شناسه زیر را با شناسه اجرای بهترین مدل خود (Run ID) جایگزین کنید
ENV MODEL_PATH=/app/mlruns/0/0cec210c37da4c0bb5e4453fe2c24b83/artifacts/model

EXPOSE 8000

# اجرای وب‌سرویس
CMD ["python", "serve_model.py"]