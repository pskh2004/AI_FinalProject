# src/features.py
import pandas as pd
from sklearn.preprocessing import StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    مهندسی ویژگی‌ها (نسخه v3) منطبق بر نام ستون‌های دیتاست خام IBM:
    - ایجاد ستون جدید تعداد کل خدمات فعال مشتری
    - دسته‌بندی ستون Tenure Months به گروه‌های زمانی مجزا
    - ایجاد ستون نسبت هزینه ماهیانه به هزینه کل پرداخت شده
    - نرمال‌سازی ستون‌های عددی اصلی به وسیله StandardScaler
    """
    df_engineered = df.copy()

    # ۱. شمارش سرویس‌های فعال مشتری (مشتمل بر اینترنت، تلفن و خدمات جانبی)
    service_keywords = [
        'Phone Service', 'Multiple Lines_Yes', 'Online Security_Yes',
        'Online Backup_Yes', 'Device Protection_Yes', 'Tech Support_Yes',
        'Streaming TV_Yes', 'Streaming Movies_Yes'
    ]

    active_cols = []
    for col in df_engineered.columns:
        if col in service_keywords or any(
                keyword in col for keyword in ['_Yes', 'Phone Service']) and col != 'Churn Value':
            active_cols.append(col)

    if active_cols:
        # جمع کردن تعداد سرویس‌های فعال
        df_engineered['Number_of_Services'] = df_engineered[active_cols].sum(axis=1)
    else:
        df_engineered['Number_of_Services'] = 0

    # ۲. دسته‌بندی مدت اشتراک (Tenure Months)
    if 'Tenure Months' in df_engineered.columns:
        df_engineered['Tenure_Group'] = pd.cut(
            df_engineered['Tenure Months'],
            bins=[-1, 12, 24, 48, 60, 100],
            labels=[1, 2, 3, 4, 5]
        ).astype(int)

    # ۳. نسبت هزینه ماهانه به کل هزینه پرداخت‌شده
    if 'Monthly Charges' in df_engineered.columns and 'Total Charges' in df_engineered.columns:
        df_engineered['Monthly_to_Total_Ratio'] = df_engineered['Monthly Charges'] / (
                    df_engineered['Total Charges'] + 1)

    # ۴. نرمال‌سازی متغیرهای عددی اصلی با StandardScaler
    num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV']
    scaler = StandardScaler()
    cols_to_scale = [col for col in num_cols if col in df_engineered.columns]

    if cols_to_scale:
        df_engineered[cols_to_scale] = scaler.fit_transform(df_engineered[cols_to_scale])

    return df_engineered