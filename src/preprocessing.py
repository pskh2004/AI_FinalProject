# src/preprocessing.py
import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    فرآیند پاک‌سازی (نسخه v2) با انطباق کامل روی ویژگی‌های دیتاست خام IBM:
    - حذف ستون‌های غیرضروری و نشت اطلاعات (Data Leakage)
    - ستون هدف نهایی: Churn Value
    - اصلاح فرمت و مقادیر گم‌شده ستون هزینه کل (Total Charges)
    - کدگذاری متغیرهای دسته‌ای به عددی (One-Hot & Binary) با حفظ دقیق Churn Value
    """
    df_cleaned = df.copy()

    # ۱. حذف ستون‌های غیرضروری، موقعیت جغرافیایی و ستون‌های مربوط به نشت اطلاعات (Data Leakage)
    drop_cols = [
        'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
        'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 'Churn Score', 'Churn Reason'
    ]
    # حذف ستون‌هایی که در دیتاست ورودی وجود دارند
    existing_drop_cols = [col for col in drop_cols if col in df_cleaned.columns]
    df_cleaned = df_cleaned.drop(columns=existing_drop_cols)

    # ۲. اصلاح ستون 'Total Charges' و مدیریت مقادیر گم‌شده
    if 'Total Charges' in df_cleaned.columns:
        # فاصله‌های خالی را به مقدار تهی تبدیل می‌کنیم
        df_cleaned['Total Charges'] = df_cleaned['Total Charges'].replace(' ', np.nan)
        df_cleaned['Total Charges'] = pd.to_numeric(df_cleaned['Total Charges'], errors='coerce')
        # مقداردهی به گم‌شده‌ها با مقدار صفر (معمولاً برای مشتریان با Tenure Months برابر با صفر)
        df_cleaned['Total Charges'] = df_cleaned['Total Charges'].fillna(0)

    # ۳. مدیریت و تثبیت نوع داده ستون هدف 'Churn Value'
    if 'Churn Value' in df_cleaned.columns:
        df_cleaned['Churn Value'] = pd.to_numeric(df_cleaned['Churn Value'], errors='coerce')
        df_cleaned = df_cleaned.dropna(subset=['Churn Value'])
        df_cleaned['Churn Value'] = df_cleaned['Churn Value'].astype(int)

    # ۴. تبدیل متغیرهای دسته‌ای به عددی (Categorical Encoding)
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()

    # حذف ستون هدف از لیست متغیرهای دسته‌ای جهت جلوگیری از تداخل و حذف ناخواسته
    if 'Churn Value' in categorical_cols:
        categorical_cols.remove('Churn Value')

    # تفکیک ستون‌ها بر اساس تعداد مقادیر منحصربه‌فرد
    binary_cols = []
    multi_cols = []
    for col in categorical_cols:
        unique_vals = df_cleaned[col].dropna().unique()
        if len(unique_vals) == 2:
            binary_cols.append(col)
        else:
            multi_cols.append(col)

    # کدگذاری متغیرهای دوحالته (مانند Gender, Senior Citizen, Partner و ...) به 0 و 1
    for col in binary_cols:
        unique_vals = list(df_cleaned[col].unique())
        df_cleaned[col] = df_cleaned[col].map({unique_vals[0]: 1, unique_vals[1]: 0})

    # کدگذاری متغیرهای چندحالته با روش One-Hot Encoding
    if multi_cols:
        df_cleaned = pd.get_dummies(df_cleaned, columns=multi_cols, drop_first=True)

    # تبدیل متغیرهای بولین خروجی get_dummies به عدد صحیح 0 و 1
    bool_cols = df_cleaned.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df_cleaned[col] = df_cleaned[col].astype(int)

    return df_cleaned