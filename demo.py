import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="NHS A&E Forecasting")

@st.cache_data
def load_and_group_data():
    # Update this path for your local environment or use the filename for Cloud deployment
    path = "finalData.csv" 
    df = pd.read_csv(path) 
    # Use dayfirst=True for UK date formats
    df['MonthYear'] = pd.to_datetime(df['MonthYear'], dayfirst=True)
    df = df.sort_values('MonthYear')
    return df

# Initialize data and global variables
df_raw = load_and_group_data()
# Narrowing data to start from 2022-01-01 to avoid pandemic outliers
df_raw = df_raw[df_raw["MonthYear"] >= "2022-01-01"]
df_filtered = pd.DataFrame()
df_actual_comparison = pd.DataFrame()

# --- 2. Main Layout (No Sidebar) ---
st.title("NHS A&E Attendances & Admissions: Multi-Model Forecasting")

st.header("Control Panel")
left_col, center_col = st.columns([2, 8])

with left_col:
    region_options = np.append(df_raw['Region'].unique(), "All Regions")
    selected_regions = st.multiselect("Select Regions", region_options, default="All Regions")
    
    if "All Regions" in selected_regions or len(selected_regions) == 0:
        df_step1 = df_raw.copy()
    else:
        df_step1 = df_raw[df_raw['Region'].isin(selected_regions)]

    available_months = sorted(df_step1['MonthYear'].unique())
    if available_months:
        selected_range = st.select_slider(
            "Historical Training Range (Data the model sees):",
            options=available_months,
            value=(available_months[0], available_months[-1]),
            format_func=lambda x: x.strftime('%Y-%m')
        )
        start_dt, end_dt = selected_range
        forecast_horizon = st.slider("Months to Predict:", 1, 24, 12)
        
        # Split data for training and post-training validation
        df_filtered = df_step1[(df_step1['MonthYear'] >= start_dt) & (df_step1['MonthYear'] <= end_dt)].copy()
        df_actual_comparison = df_step1[df_step1['MonthYear'] > end_dt].copy()
    else:
        st.error("No data available.")
        st.stop()

    run_forecast = st.button("Run Forecast Analysis")
    st.markdown("---")

with center_col:    
    # --- 3. Historical KPIs ---
    if not df_filtered.empty:
        st.subheader("Historical Performance Metrics (Aggregated)")
        kpi_vals = df_filtered.agg({
            'Type 1': 'sum', 'Type 2': 'sum', 'Type 3': 'sum',
            'Total Attendances': 'sum', 'Total Emergency Admissions': 'sum',
            'Within_4h_%': 'mean'
        })

        kcols = st.columns(7)
        kcols[0].metric("Total Attendances", f"{int(kpi_vals['Total Attendances']):,}")
        kcols[1].metric("Type 1", f"{int(kpi_vals['Type 1']):,}")
        kcols[2].metric("Type 2", f"{int(kpi_vals['Type 2']):,}")
        kcols[3].metric("Type 3", f"{int(kpi_vals['Type 3']):,}")
        kcols[4].metric("Admissions", f"{int(kpi_vals['Total Emergency Admissions']):,}")
        kcols[5].metric("Within 4h %", f"{round(float(kpi_vals['Within_4h_%']), 2)}%")
        kcols[6].metric("Over 4h %", f"{round(100 - float(kpi_vals['Within_4h_%']), 2)}%")

        # --- 3b. Baseline Trend Visualization ---
        st.subheader("Historical Attendance Trend")
        baseline_data = df_filtered.groupby('MonthYear')['Total Attendances'].sum().reset_index()
        
        fig_base, ax_base = plt.subplots(figsize=(14, 4))
        sns.lineplot(x='MonthYear', y='Total Attendances', data=baseline_data, color='black', marker='o')
        ax_base.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.2f}M'))
        ax_base.set_ylabel("Attendances (Millions)")
        plt.xticks(rotation=45)
        st.pyplot(fig_base)

    st.markdown("---")

    # --- 4. Forecasting Logic ---
    if run_forecast:
        model_type = "ARIMA" if forecast_horizon <= 3 else "Prophet"
        with st.spinner(f'Calculating Triple-Model Forecast & Admissions Regression...'):
            
            # --- LINEAR REGRESSION TRAINING ---
            lr_train_data = df_filtered.groupby('MonthYear').agg({
                'Total Attendances': 'sum',
                'Total Emergency Admissions': 'sum'
            }).reset_index()
            
            LRModel = LinearRegression()
            X_lr = lr_train_data[['Total Attendances']]
            y_lr = lr_train_data['Total Emergency Admissions']
            LRModel.fit(X_lr, y_lr)
            
            # --- TIME SERIES FORECASTING (ATTENDANCES) ---
            targets = ['Type 1', 'Type 2', 'Type 3']
            agg_train = df_filtered.groupby('MonthYear')[targets].sum().reset_index()
            
            forecast_parts = []
            target_predictions = {} 
            
            for t in targets:
                if model_type == "ARIMA":
                    arimadata = agg_train.rename(columns={'MonthYear': 'ds', t: 'y'})
                    arimadata = arimadata.set_index('ds')['y'].asfreq('MS')
                    modelarimafit = sm.tsa.ARIMA(arimadata, order=(1,1,1)).fit()
                    forecast_res = modelarimafit.forecast(steps=forecast_horizon)
                    forecast = pd.DataFrame({'ds': forecast_res.index, 'yhat': forecast_res.values})
                else:
                    m_df = agg_train[['MonthYear', t]].rename(columns={'MonthYear': 'ds', t: 'y'})
                    m = Prophet(interval_width=0.95, yearly_seasonality=True).fit(m_df)
                    future = m.make_future_dataframe(periods=forecast_horizon, freq='MS')
                    forecast = m.predict(future)
                
                pred_df = forecast[forecast['ds'] > end_dt].copy()
                target_predictions[t] = pred_df['yhat'].sum()
                pred_df['target'] = t
                forecast_parts.append(pred_df[['ds', 'yhat', 'target']])
            
            combined = pd.concat(forecast_parts)
            total_forecast = combined.groupby('ds')['yhat'].sum().reset_index()
            total_forecast.rename(columns={'yhat': 'ForecastAttendances'}, inplace=True)

            # --- PREDICT ADMISSIONS (LINEAR REGRESSION) ---
            X_future = total_forecast[['ForecastAttendances']].rename(columns={'ForecastAttendances': 'Total Attendances'})
            total_forecast['PredictedAdmissions'] = LRModel.predict(X_future).astype(int)

            # --- 5. Predicted KPIs ---
            st.subheader(f"Forecast Summary: Next {forecast_horizon} Months")
            pkpi = st.columns(5)
            total_pred_attend = total_forecast['ForecastAttendances'].sum()
            total_pred_admit = total_forecast['PredictedAdmissions'].sum()
            
            pkpi[0].metric("Total Pred. Attendances", f"{int(total_pred_attend):,}")
            pkpi[1].metric("Total Pred. Admissions", f"{int(total_pred_admit):,}")
            pkpi[2].metric("Type 1 Pred.", f"{int(target_predictions['Type 1']):,}")
            pkpi[3].metric("Type 2 Pred.", f"{int(target_predictions['Type 2']):,}")
            pkpi[4].metric("Type 3 Pred.", f"{int(target_predictions['Type 3']):,}")

            # --- 6. Evaluation Metrics ---
            if not df_actual_comparison.empty:
                st.markdown("---")
                st.subheader("Model Validation: Accuracy Metrics")
                actuals_agg = df_actual_comparison.groupby('MonthYear')[targets].sum().reset_index()
                actuals_agg['y_true'] = actuals_agg[targets].sum(axis=1)
                
                eval_df = pd.merge(actuals_agg[['MonthYear', 'y_true']], total_forecast, left_on='MonthYear', right_on='ds')
                
                if not eval_df.empty:
                    y_true, y_pred = eval_df['y_true'], eval_df['ForecastAttendances']
                    mae = mean_absolute_error(y_true, y_pred)
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = np.sqrt(mse)
                    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.info(f"**MAE** \n {mae:,.0f}")
                    m2.info(f"**MSE** \n {mse:,.2e}")
                    m3.info(f"**RMSE** \n {rmse:,.0f}")
                    m4.info(f"**MAPE** \n {mape:.2f}%")

            # --- 7. Attendance Chart (CONNECTED) ---
            st.subheader(f"Attendance Analysis: Actuals vs. {model_type} Predictions")
            fig1, ax1 = plt.subplots(figsize=(14, 5))
            hist_attend = agg_train.copy()
            hist_attend['Total'] = hist_attend[targets].sum(axis=1)
            
            conn_attend = pd.DataFrame({'ds': [end_dt], 'ForecastAttendances': [hist_attend['Total'].iloc[-1]]})
            plot_attend = pd.concat([conn_attend, total_forecast[['ds', 'ForecastAttendances']]])

            sns.lineplot(x='MonthYear', y='Total', data=hist_attend, color='black', marker='o', label='Historical')
            if not df_actual_comparison.empty:
                comp_attend = df_actual_comparison.groupby('MonthYear')[targets].sum().reset_index()
                comp_attend['Total'] = comp_attend[targets].sum(axis=1)
                sns.lineplot(x='MonthYear', y='Total', data=comp_attend, color='green', label='Actual (Post-Training)')
            
            sns.lineplot(x='ds', y='ForecastAttendances', data=plot_attend, color='red', linestyle='--', label='Predicted')
            ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.2f}M'))
            st.pyplot(fig1)

            # --- 8. Admissions Chart (CONNECTED) ---
            st.subheader("Emergency Admissions Analysis: Actuals vs. Linear Regression Predictions")
            fig2, ax2 = plt.subplots(figsize=(14, 5))
            hist_admit = df_filtered.groupby('MonthYear')['Total Emergency Admissions'].sum().reset_index()
            
            conn_admit = pd.DataFrame({'ds': [end_dt], 'PredictedAdmissions': [hist_admit['Total Emergency Admissions'].iloc[-1]]})
            plot_admit = pd.concat([conn_admit, total_forecast[['ds', 'PredictedAdmissions']]])

            sns.lineplot(x='MonthYear', y='Total Emergency Admissions', data=hist_admit, color='black', marker='o', label='Historical Admissions')
            if not df_actual_comparison.empty:
                comp_admit = df_actual_comparison.groupby('MonthYear')['Total Emergency Admissions'].sum().reset_index()
                sns.lineplot(x='MonthYear', y='Total Emergency Admissions', data=comp_admit, color='pink', label='Actual Admissions')
            
            sns.lineplot(x='ds', y='PredictedAdmissions', data=plot_admit, color='red', marker='o', label='Predicted Admissions (LR)')
            plt.legend()
            st.pyplot(fig2)

    else:
        st.info("Select regions and training range, then click Run Forecast Analysis.")

    # --- 9. Table ---
    with st.expander("View Filtered Data Table"):
        if not df_filtered.empty:
            table_display = df_filtered.copy()
            table_display['MonthYear'] = table_display['MonthYear'].dt.strftime('%Y-%m')
            st.dataframe(table_display, use_container_width=True)
