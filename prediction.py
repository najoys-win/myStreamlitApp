import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import streamlit as st

def fetch_predict_plot(selected_channel, collection, df):
    if 'display_plot' not in st.session_state:
        st.session_state.display_plot = False

    if st.sidebar.button("Predict Future Views"):
        prediction_data = list(collection.find({'Channel': selected_channel}))

        if prediction_data:
            last_date = df['PublishedAt'].max()
            future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=len(prediction_data), freq='D')
            future_views = [item['FutureViewing'] for item in prediction_data]
            
            st.session_state.plot_data = (future_dates, future_views)
            st.session_state.display_plot = True
        else:
            st.error(f"No prediction data available for {selected_channel}.")

    if st.session_state.display_plot:
        future_dates, future_views = st.session_state.plot_data
        st.markdown("## Future View Prediction")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(future_dates, future_views, marker='o', linestyle='-', linewidth=2, label="Predicted Views", color="#00aaff")
        ax.set_title(f'Predicted Views for {selected_channel}', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Predicted View Count', fontsize=14)
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x / 1e6)}M"))
        date_form = DateFormatter("%Y-%m-%d")
        ax.xaxis.set_major_formatter(date_form)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.set_facecolor("#00172B")
        fig.patch.set_facecolor("#00172B")
        st.pyplot(fig)
