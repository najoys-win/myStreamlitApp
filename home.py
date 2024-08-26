import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from pymongo import MongoClient
import hashlib
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import isodate
from dateutil.parser import parse
from matplotlib.dates import DateFormatter
from pandas import Timestamp
import numpy as np



# Dashboard setup
st.set_page_config(page_title="YouTube Channel Analytics", layout="wide")

# Display a title in the app content
st.title("YouTube Channel Analytics Dashboard")

def load_data(path):
    return pd.read_csv(path, parse_dates=['PublishedAt'], usecols=['Title', 'PublishedAt', 'ViewCount', 'LikeCount', 'CommentCount', 'URL'])


#base_path = os.getenv('DATA_PATH', 'D:/Master Project/Dissertation')
base_path = os.getenv('DATA_PATH', './data')
data_paths = {
    'Technology': {'Linus Tech Tips': f'{base_path}/Linus Tech Tips.csv', 'Code Nust': f'{base_path}/code nust.csv'},
    'News': {'BBC News': f'{base_path}/BBC NEWS.csv', 'CNN News': f'{base_path}/CNN NEWS.csv'},
    'Vlog': {'NicoleRafiee': f'{base_path}/Nicole Rafiee.csv', 'Sydney Serena': f'{base_path}/Sydney Serena.csv'},
    'Mukbang': {'Hamzy': f'{base_path}/Hamzy.csv', 'ZCM ASMR': f'{base_path}/ZCM ASMR.csv'},
    'Gaming': {'NinJa': f'{base_path}/NinJa.csv', 'Failboat': f'{base_path}/Failboat.csv'}
}


def format_number(num):
    try:
        num = float(num)
    except (ValueError, TypeError):
        return "N/A"
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(int(num))

def prepare_table(data):
    data['ViewCount'] = data['ViewCount'].apply(format_number)
    data['LikeCount'] = data['LikeCount'].apply(format_number)
    data['URL'] = data['URL'].apply(lambda x: f'<a href="{x}" target="_blank">Watch Video</a>')
    return data.to_html(escape=False, index=False)

def plot_data(data, column, ax, title):
    data.set_index('PublishedAt')[column].plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel(column)
    ax.grid(True)


selected_category = st.sidebar.selectbox('Select Category', list(data_paths.keys()))
selected_channel = st.sidebar.selectbox('Select Channel', list(data_paths[selected_category].keys()))
df_path = data_paths[selected_category][selected_channel]

try:
    df = load_data(df_path)
except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
    st.stop()

date_range = st.sidebar.date_input("Define the timeframe: ", [])

if date_range:
    start_date, end_date = date_range
    # Ensure both dates are timezone-aware and match the DataFrame's timezone.
    start_date = Timestamp(start_date).tz_localize('UTC')
    end_date = Timestamp(end_date).tz_localize('UTC')
    filtered_data = df[(df['PublishedAt'] >= start_date) & (df['PublishedAt'] <= end_date)]
    
    if not filtered_data.empty:
        st.markdown("## Summary Metrics")
        total_views = format_number(filtered_data['ViewCount'].sum())
        total_likes = format_number(filtered_data['LikeCount'].sum())
        total_comments = format_number(filtered_data['CommentCount'].sum())
        total_videos = format_number(filtered_data.shape[0])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Views", total_views)
        col2.metric("Total Likes", total_likes)
        col3.metric("Total Comments", total_comments)
        col4.metric("Total Videos", total_videos)

        # Create two columns for the table and chart
        col1, col2 = st.columns(2)
        
                # Display the table in the first column
        # Display the table in the first column
        with col1:
            st.markdown("## Video Performance Over Time")
            fig, ax = plt.subplots()
            # Applying theme colors from config.toml
            background_color = "#00172B"  # Dark blue
            primary_color = "#154C79"     # Lighter blue for the line
            text_color = "#ffff"        # White text for better readability
            # Set plot and background colors
            ax.set_facecolor(background_color)
            ax.figure.set_facecolor(background_color)
            # Plotting the data with theme primary color
            filtered_data.set_index('PublishedAt')['ViewCount'].plot(ax=ax, color=primary_color)
            # Setting title and labels with the theme text color
            ax.set_title('Video ViewCount Over Time', color=text_color)
            ax.set_xlabel('Date', color=text_color)
            ax.set_ylabel('ViewCount', color=text_color)
            # Customizing tick parameters
            ax.tick_params(axis='x', colors=text_color)
            ax.tick_params(axis='y', colors=text_color)
            # Adding grid with a lighter tone
            #ax.grid(True, color="#639700")  # Using secondary background color for grid
            st.pyplot(fig)

        with col2:
            st.markdown("## Video Performance Over Time")
            fig, ax = plt.subplots()
            # Applying theme colors from config.toml
            background_color = "#00172B"  # Dark blue
            primary_color = "#154C79"     # Lighter blue for the line
            text_color = "#ffff"        # White text for better readability
            # Set plot and background colors
            ax.set_facecolor(background_color)
            ax.figure.set_facecolor(background_color)
            # Plotting the data with theme primary color
            filtered_data.set_index('PublishedAt')['LikeCount'].plot(ax=ax, color=primary_color)
            # Setting title and labels with the theme text color
            ax.set_title('Video LikeCount Over Time', color=text_color)
            ax.set_xlabel('Date', color=text_color)
            ax.set_ylabel('LikeCount', color=text_color)
            # Customizing tick parameters
            ax.tick_params(axis='x', colors=text_color)
            ax.tick_params(axis='y', colors=text_color)
            # Adding grid with a lighter tone
            #ax.grid(True, color="#639700")  # Using secondary background color for grid
            st.pyplot(fig)

        sorted_data = filtered_data.sort_values(by='ViewCount', ascending=False).head(5)
        html = prepare_table(sorted_data)
        st.markdown("## Top 5 Videos", unsafe_allow_html=True)
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.error("No data available for the selected date range.")
        nearest_date = df['PublishedAt'].min()
        st.info(f"Try starting from {nearest_date.strftime('%Y-%m-%d')}")


# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client['mdb']
collection = db['future']


import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


# Check if the data and plot need to be generated or displayed
if 'display_plot' not in st.session_state:
    st.session_state.display_plot = False

# Button to predict future views
if st.sidebar.button("Predict Future Views"):
    # Fetch prediction data from MongoDB for the selected channel
    prediction_data = list(collection.find({'Channel': selected_channel}))

    if prediction_data:
        # Generate future dates dynamically based on the number of predictions
        last_date = df['PublishedAt'].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=len(prediction_data), freq='D')
        
        # Extract the future viewing predictions
        future_views = [item['FutureViewing'] for item in prediction_data]
        
        # Store the plot data in session state
        st.session_state.plot_data = (future_dates, future_views)
        st.session_state.display_plot = True  # Flag to display the plot
    else:
        st.error(f"No prediction data available for {selected_channel}.")

# Display the plot if flag is set
if st.session_state.display_plot:
    future_dates, future_views = st.session_state.plot_data
    st.markdown("## Future View Prediction")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(future_dates, future_views, marker='o', linestyle='-', linewidth=2, label="Predicted Views", color="#00aaff")
    ax.set_title(f'Predicted Views for {selected_channel}', color="#ffff", fontsize=16)
    ax.set_xlabel('Date', color="#ffff", fontsize=14)
    ax.set_ylabel('Predicted View Count (Millions)', color="#ffff", fontsize=14)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{int(x / 1e6)}M"))
    date_form = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    ax.grid(True, color="#555555", linestyle='--', linewidth=0.5)
    ax.tick_params(axis='x', colors="#ffff", which='both')
    ax.tick_params(axis='y', colors="#ffff", which='both')
    ax.set_facecolor("#00172B")
    fig.patch.set_facecolor("#00172B")
    st.pyplot(fig)



# MongoDB connection setup
client = MongoClient("mongodb+srv://htet3win:htet3winforyoutube@youtubevideoanalysis.cbkiow5.mongodb.net/?retryWrites=true&w=majority&appName=YouTubeVideoAnalysis")
db = client.usersdata
users = db.auth

# Initialize YouTube API
API_KEY = 'AIzaSyAB7IhXukv3OUesHkDQnWTLB-8lIZqON1I'
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_sortable_cols(df):
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    datetime_columns = ['Published At']
    return numeric_columns + datetime_columns
    
# Apply formatting to the numeric columns and prepare the table
def prepare_tb(df):
    df['View Count'] = df['View Count'].apply(format_number)
    df['Like Count'] = df['Like Count'].apply(format_number)
    df['URL'] = df['URL'].apply(lambda x: f'<a href="{x}" target="_blank">Watch Video</a>')
    return df.to_html(escape=False, index=False)

def create_user(username, password):
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    users.insert_one({"username": username, "password": hashed_pw})

def check_user(username, password):
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    user = users.find_one({"username": username, "password": hashed_pw})
    return user is not None

def get_channel_id_by_name(channel_name):
    try:
        request = youtube.search().list(
            part='snippet',
            q=channel_name,
            type='channel',
            maxResults=1
        )
        response = request.execute()
        return response['items'][0]['snippet']['channelId']
    except HttpError as e:
        st.error(f"An HTTP error {e.resp.status} occurred: {e.content}")
        return None

import pandas as pd  # make sure to import pandas at the beginning of your script

def get_videos_in_playlist(playlist_id, start_date, end_date):
    videos = []
    request = youtube.playlistItems().list(
        part='snippet,contentDetails',
        playlistId=playlist_id,
        maxResults=50  # Increased to fetch more data per API call
    )
    
    while request is not None:
        response = request.execute()
        for item in response['items']:
            video_date = pd.to_datetime(item['snippet']['publishedAt']).date()
            # Ensure start_date and end_date are also datetime.date objects
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date).date()
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date).date()

            if start_date <= video_date <= end_date:
                videos.append(item['contentDetails']['videoId'])
        request = youtube.playlistItems().list_next(request, response)
    
    return videos
    
import re

def convert_duration_to_iso8601(duration):
    if not duration:
        return 'PT0S'  # Default to 0 seconds if duration is empty or malformed

    pattern = re.compile(r'(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?')
    match = pattern.match(duration.strip())
    hours, minutes, seconds = match.groups(default='0')
    
    # Correctly formatting the ISO 8601 duration string
    iso_duration = 'PT'
    if hours != '0':
        iso_duration += f'{hours}H'
    if minutes != '0':
        iso_duration += f'{minutes}M'
    if seconds != '0':
        iso_duration += f'{seconds}S'
    if iso_duration == 'PT':
        iso_duration = 'PT0S'  # Default to 0 seconds if no time components are found

    return iso_duration


def get_channel_statistics(channel_id):
    request = youtube.channels().list(
        part='snippet,statistics,contentDetails',
        id=channel_id
    )
    response = request.execute()
    
    if response['items']:
        item = response['items'][0]
        snippet = item['snippet']
        statistics = item['statistics']
        
        uploads_playlist_id = item['contentDetails']['relatedPlaylists']['uploads']
        videos = get_videos_in_playlist(uploads_playlist_id, "2000-01-01", "2100-01-01")
        video_details = get_video_details(videos)

        average_length_seconds = np.mean([
            isodate.parse_duration(convert_duration_to_iso8601(video.get('Duration', '0s'))).total_seconds()
            for video in video_details
        ]) if video_details else 0

        formatted_subscriber_count = format_number(statistics.get('subscriberCount'))
        
        channel_info = {
            'Channel Name': snippet['title'],
            'Subscribers': formatted_subscriber_count,
            'Creation Date': snippet['publishedAt'],
            'Total Videos': statistics.get('videoCount', 'N/A'),
            'Category': snippet.get('categoryId', 'Unknown'),
            'Average Video Length': f'{int(average_length_seconds // 60)}m {int(average_length_seconds % 60)}s' if average_length_seconds else 'N/A',
            'Location': snippet.get('country', 'N/A')
        }
        return channel_info
    return {}




def get_video_details(video_ids):
    max_ids_per_request = 10  # Corrected variable name
    videos_info = []

    # Split the video_ids into chunks of 50
    for i in range(0, len(video_ids), max_ids_per_request):
        batch_ids = video_ids[i:i + max_ids_per_request]
        request = youtube.videos().list(
            part='snippet,statistics,contentDetails',
            id=','.join(batch_ids)
        )
        try:
            response = request.execute()
            for item in response['items']:
                duration = isodate.parse_duration(item['contentDetails']['duration'])
                minutes = int(duration.total_seconds() // 60)
                seconds = int(duration.total_seconds() % 60)
                formatted_duration = f'{minutes}m {seconds}s'

                # Extracting publishedAt date
                published_at = item['snippet']['publishedAt']

                videos_info.append({
                    'Title': item['snippet']['title'],
                    'Published At': published_at,
                    'View Count': item['statistics'].get('viewCount', 'N/A'),
                    'Like Count': item['statistics'].get('likeCount', 'N/A'),
                    'Comment Count': item['statistics'].get('commentCount', 'N/A'),
                    'URL': f"https://www.youtube.com/watch?v={item['id']}",
                    'Duration': formatted_duration
                })
        except HttpError as e:
            st.error(f"Failed to retrieve video details: {e}")
            break

    return videos_info

def display_channel_info(channel_info):
    df = pd.DataFrame([channel_info])
    return df.to_html(index=False, escape=False)

def show_videos(video_details):
    # This function would use the logic similar to what was discussed earlier to generate HTML content
    html_content = generate_video_display_html(video_details)
    st.markdown(html_content, unsafe_allow_html=True)


def show_analysis():
    channel_name = st.text_input("Enter the Channel Name:")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    
    if st.button("Analyze") and channel_name:
        channel_id = get_channel_id_by_name(channel_name)
        if channel_id:
            channel_details = youtube.channels().list(part='snippet,contentDetails', id=channel_id).execute()
            
            if not channel_details.get('items'):
                st.error("Channel details not found. Please check the channel name or ID.")
                return
            
            channel_title = channel_details['items'][0]['snippet']['title']
            st.subheader(f"Analysis - YouTube Channel Details for {channel_title}")
            uploads_playlist_id = channel_details['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            
            # Fetch all videos for statistics calculations and then filter by date for display
            all_video_ids = get_videos_in_playlist(uploads_playlist_id, "2000-01-01", "2100-01-01")
            all_video_details = get_video_details(all_video_ids)
            
            # Now fetch and display channel statistics
            channel_info = get_channel_statistics(channel_id)
            html_info = display_channel_info(channel_info)
            st.markdown(html_info, unsafe_allow_html=True)
            
            # Filter videos by user specified date range
            video_details = [video for video in all_video_details if start_date <= pd.to_datetime(video['Published At']).date() <= end_date]
            
            if video_details:
                prepare_and_display_data(pd.DataFrame(video_details))
                #generate_video_display_html(video_details)  # Plot the last three updated videos
                

            else:
                st.write("No videos found in the selected timeframe.")
        else:
            st.error("Failed to fetch channel details. Please check the channel name.")

import matplotlib.pyplot as plt




def prepare_and_display_data(df):
    # Data preparation as described previously
    df['Published At'] = pd.to_datetime(df['Published At'], errors='coerce', format='%Y-%m-%dT%H:%M:%SZ')
    df.dropna(subset=['Published At'], inplace=True)
    df.sort_values('Published At', inplace=True)

    # Numeric conversions
    df['View Count'] = pd.to_numeric(df['View Count'], errors='coerce')
    df['Like Count'] = pd.to_numeric(df['Like Count'], errors='coerce')
    df['Comment Count'] = pd.to_numeric(df['Comment Count'], errors='coerce')

    # Metric calculations
    total_views = format_number(df['View Count'].sum())
    total_likes = format_number(df['Like Count'].sum())
    total_comments = format_number(df['Comment Count'].sum())
    total_videos = format_number(len(df))  # Number of videos might not need 'M' or 'K' suffix

    # Display metrics
    st.markdown("## Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Views", total_views)
    col2.metric("Total Likes", total_likes)
    col3.metric("Total Comments", total_comments)
    col4.metric("Total Videos", total_videos)


    # Continue with plotting and other data handling...


    sortable_cols = get_sortable_cols(df)
    selected_sort = st.sidebar.selectbox('Sort Videos By', sortable_cols)
    sort_data = df.sort_values(by=selected_sort, ascending=False).head(10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(df['Published At'], df['View Count'], marker='o', linestyle='-', color='cyan')
    ax2.plot(df['Published At'], df['Like Count'], marker='o', linestyle='-', color='lightgreen')
    background_color = "#00172B"
    text_color = "#ffff"
    ax1.set_facecolor(background_color)
    ax2.set_facecolor(background_color)
    ax1.figure.set_facecolor(background_color)
    ax2.figure.set_facecolor(background_color)
    ax1.set_title('Video LikeCount Over Time', color=text_color)
    ax1.set_xlabel('Date', color=text_color)
    ax1.set_ylabel('LikeCount', color=text_color)
    # Customizing tick parameters
    ax1.tick_params(axis='x', colors=text_color)
    ax1.tick_params(axis='y', colors=text_color)
    ax2.set_title('Video ViewCount Over Time', color=text_color)
    ax2.set_xlabel('Date', color=text_color)
    ax2.set_ylabel('ViewCount', color=text_color)
    # Customizing tick parameters
    ax2.tick_params(axis='x', colors=text_color)
    ax2.tick_params(axis='y', colors=text_color)
 
    plt.tight_layout()
    st.pyplot(fig)

    # Display formatted table
    html = prepare_tb(sort_data.copy())  # Use copy to avoid SettingWithCopyWarning
    st.markdown("## Top 10 Videos", unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)




def main():
    #st.title("YouTube Channel Analytics")

    # Login and Registration tabs
    menu = ["Login", "SignUp", "Analysis"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Analysis":
        if "logged_in" in st.session_state and st.session_state["logged_in"]:
            show_analysis()
        else:
            st.warning("Please login to access this feature.")

    elif choice == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.button("Login"):
            if check_user(username, password):
                st.session_state["logged_in"] = True
                st.success(f"Logged In as {username}")
            else:
                st.error("Incorrect Username/Password")

    elif choice == "SignUp":
        new_username = st.sidebar.text_input("New Username")
        new_password = st.sidebar.text_input("New Password", type='password')
        if st.sidebar.button("Create Account"):
            create_user(new_username, new_password)
            st.success("You have successfully created an account!")
            st.info("Go to the Login Menu to login")

if __name__ == "__main__":
    main()

