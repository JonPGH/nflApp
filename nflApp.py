import streamlit as st
import pandas as pd, math
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Define the correct password (replace with your desired password)
#st.markdown("<h1>Enter Password to Access Slate Analysis Tool",unsafe_allow_html=True)
CORRECT_PASSWORD = "foster"
CORRECT_PASSWORD2 = '1'

# Function to check password
def check_password():
    def password_entered():
        if (st.session_state["password"] == CORRECT_PASSWORD) or (st.session_state["password"] == CORRECT_PASSWORD2):
            st.session_state.authenticated = True
            del st.session_state["password"]  # Clear password from session state
        else:
            st.error("Incorrect password. Please try again.")
    
    if not st.session_state.authenticated:
        st.text_input("Enter Password (new password in resource glossary 7/18/2025)", type="password", key="password", on_change=password_entered)
        return False
    return True

# Main app content (only displayed if authenticated)
if check_password():

    # Set page configuration
    st.set_page_config(page_title="Follow The Money Fantasy Football App", layout="wide")

    st.markdown(
        """
        <style>
        /* Base styling */
        .stApp {
            background-color: white; /* #F5F6F5; Light gray background for a clean look */
            font-family: 'Roboto', sans-serif; /* Modern font */
        }
        h1, h2, h3, h4 {
            color: #003087; /* Navy blue for headers, MLB-inspired */
            font-weight: 700;
        }
        .sidebar .sidebar-content {
            background-color: #FFFFFF;
            border-right: 1px solid #E0E0E0;
        }
        /* Card styling for player profiles */
        .player-card {
            background-color: #FFFFFF;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 20px;
        }
        /* Table styling */
        .dataframe {
            border-collapse: collapse;
            width: 100%;
        }
        .dataframe th, .dataframe td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #E0E0E0;
        }
        .dataframe tr:nth-child(even) {
            background-color: #F9F9F9; /* Alternating row colors */
        }
        /* Button and selectbox styling */
        .stSelectbox, .stRadio > div {
            background-color: #FFFFFF;
            border-radius: 5px;
            padding: 5px;
            border: 1px solid #E0E0E0;
        }
        /* Responsive design */
        @media (max-width: 768px) {
            .player-card {
                padding: 10px;
            }
            h1 {
                font-size: 24px;
            }
            h2 {
                font-size: 20px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
        <style>
        .stDataFrame {
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            overflow: hidden;
        }
        .dataframe {
            width: 100%;
            font-size: 14px;
        }
        .dataframe th {
            background-color: #f4f4f4;
            color: #333;
            font-weight: bold;
            padding: 10px;
            text-align: left;
            border-bottom: 2px solid #ddd;
        }
        .dataframe td {
            padding: 8px;
            border-bottom: 1px solid #eee;
        }
        .dataframe tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .dataframe tr:hover {
            background-color: #f1f1f1;
        }
        .stSelectbox, .stTextInput, .stRadio > div {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 5px;
        }
        h3 {
            color: #1f77b4;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .stMarkdown p {
            color: #666;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("""
        <style>
        .stDataFrame {
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            overflow: hidden;
        }
        .dataframe {
            width: 100%;
            font-size: 14px;
        }
        .dataframe th {
            background-color: #f4f4f4;
            color: #333;
            font-weight: bold;
            padding: 10px;
            text-align: left;
            border-bottom: 2px solid #ddd;
        }
        .dataframe td {
            padding: 8px;
            border-bottom: 1px solid #eee;
        }
        .dataframe tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .dataframe tr:hover {
            background-color: #f1f1f1;
        }
        .stSelectbox, .stTextInput, .stRadio > div {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 5px;
        }
        h3 {
            color: #1f77b4;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .stMarkdown p {
            color: #666;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    @st.cache_data
    def load_data():
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, 'Data')
        adp_data = pd.read_csv(f'{file_path}/ADP_Dashboard.csv')
        logo = "{}/Logo.png".format(file_path)
        season_proj = pd.read_csv(f'{file_path}/JA_Season_Projections.csv')

        return logo, adp_data, season_proj
    logo, adp_data, season_proj = load_data()
    season_proj['Proj FPts'] = 0
    
    st.sidebar.image(logo, width=250)  # Added logo to sidebar
    st.sidebar.title("Fantasy Football Resources")
    tab = st.sidebar.radio("Select View", ["ADP Data", "Season Projections"], help="Choose a Page")
    
    if "reload" not in st.session_state:
        st.session_state.reload = False

    if st.sidebar.button("Reload Data"):
        st.session_state.reload = True
        st.cache_data.clear()  # Clear cache to force reload

    # Main content
    st.markdown(f"<center><h1>Follow The Money Fantasy Football Web App</h1></center>", unsafe_allow_html=True)
    if tab == "ADP Data":
        adp_data = adp_data.sort_values(by='Date')
        adp_min = int(adp_data['ADP'].min())
        adp_max = int(adp_data['ADP'].max())
        col1,col2 = st.columns([1,1])
        with col1:
            adp_range = st.slider('ADP Range', adp_min, adp_max, (adp_min, adp_max))
        with col2:
            pos_list = ['All'] + ['Flex'] + sorted(adp_data['Pos'].dropna().unique().tolist())
            pos_filter = st.selectbox(
                label="Filter by Pos",options=pos_list,
                index=0,help="Select a position to filter ADP data", key="pos_filter")        
        
        st.write("<hr>",unsafe_allow_html=True)
        
        col1, col2 = st.columns([1,1])
        
        with col1:
            st.markdown("<h3>Main Data</h3>",unsafe_allow_html=True)
            all_adp = adp_data.groupby(['Player','Team','Pos'],as_index=False)['ADP'].mean()
            all_adp = all_adp.round(1)
            all_adp = all_adp.sort_values(by='ADP')
            
            # ADP Range Filter
            show_adp = all_adp[(all_adp['ADP']>=adp_range[0])&(all_adp['ADP']<=adp_range[1])]

            if pos_filter == 'All':
                st.dataframe(show_adp,hide_index=True,width=470,height=750)
            elif pos_filter == 'Flex':
                st.dataframe(show_adp[show_adp['Pos'].isin(['RB','WR','TE'])],hide_index=True,width=470,height=750)
            else:
                st.dataframe(show_adp[show_adp['Pos']==pos_filter],hide_index=True,width=470,height=750)
        
        with col2:
            st.markdown("<h3>Trends</h3>",unsafe_allow_html=True)
            all_adp = adp_data.groupby(['Player','Team','Pos'],as_index=False)['ADP'].mean()
            all_adp = all_adp.round(1)
            all_adp = all_adp.sort_values(by='ADP')

            all_dates = adp_data['Date'].unique()
            l7dates = all_dates[-7:]

            l7_adp = adp_data[adp_data['Date'].isin(l7dates)].groupby(['Player','Team','Pos'],as_index=False)['ADP'].mean()
            l7_adp = l7_adp.round(1)
            l7_adp = l7_adp.sort_values(by='ADP')
            l7_adp.columns=['Player','Team','Pos','L7']

            l7_merge = pd.merge(all_adp,l7_adp,on=['Player','Team','Pos'],how='left')
            l7_merge['L7 Change'] = round(l7_merge['ADP']-l7_merge['L7'],1)
            
            show_l7_adp = l7_merge[(l7_merge['ADP']>=adp_range[0])&(l7_merge['ADP']<=adp_range[1])]
            show_l7_adp = show_l7_adp.sort_values(by='L7 Change')
            if pos_filter == 'All':
                st.dataframe(show_l7_adp,hide_index=True,width=470,height=750)
            elif pos_filter == 'Flex':
                st.dataframe(show_l7_adp[show_l7_adp['Pos'].isin(['RB','WR','TE'])],hide_index=True,width=470,height=750)
            else:
                st.dataframe(show_l7_adp[show_l7_adp['Pos']==pos_filter],hide_index=True,width=470,height=750)

    elif tab == "Season Projections":
        st.markdown(f"<center><h3>Season Long Projections</h3></center>", unsafe_allow_html=True)
        col1,col2,col3,col4,col5 = st.columns([1,1,1,1,9])
        with col1:
            show_qb = st.checkbox('QB', value=True)
        with col2:
            show_rb = st.checkbox('RB', value=True)
        with col3:
            show_wr = st.checkbox('WR', value=True)
        with col4:
            show_te = st.checkbox('TE', value=True)

        pos_selected = []
        if show_qb is True:
            pos_selected.append('QB')
        if show_rb is True:
            pos_selected.append('RB')
        if show_wr is True:
            pos_selected.append('WR')
        if show_te is True:
            pos_selected.append('TE')
        
        if 'QB' not in pos_selected:
            show_cols = ['Player','Team','Pos','Proj FPts','Rush Att','Rush Yards','Rush TD','Rec','Rec Yards','Rec TD']
        elif ('QB' in pos_selected) & ('RB' not in pos_selected) & ('WR' not in pos_selected) & ('TE' not in pos_selected):
            show_cols = ['Player','Team','Pos','Proj FPts','Pass Att','Pass Yards','Int','Pass TD','Rush Att','Rush Yards','Rush TD']
        else:
            show_cols = ['Player','Team','Pos','Proj FPts','Pass Att','Pass Yards','Int','Pass TD','Rush Att','Rush Yards','Rush TD','Rec','Rec Yards','Rec TD']

        show_proj = season_proj[season_proj['Pos'].isin(pos_selected)]
        
        a_col1, a_col2 = st.columns([1,4])
        with a_col1:
            b_col1, b_col2 = st.columns([1,1])
            with b_col1:
                pass_yards_per_point = st.text_input("Pass Yds/Pt:", value=25)
                pts_passtd = st.text_input("Pass TD:", value=4)
                pts_passint = st.text_input("Pass Int:", value=-1)
            with b_col2:
                pts_rushrecyd = st.text_input("Rush/Rec Yds/Pt:", value=10)
                pts_rushrectd = st.text_input("Rush/Rec TD:", value=6)
                pts_rec = st.text_input("Pts/Rec:", value=1)
            
            pass_yards_per_point = float(pass_yards_per_point)
            pts_passtd = float(pts_passtd)
            pts_passint = float(pts_passint)
            pts_rushrecyd = float(pts_rushrecyd)
            pts_rushrectd = float(pts_rushrectd)
            pts_rec = float(pts_rec)

        with a_col2:
            show_proj['Proj FPts'] = (show_proj['Pass Yards']/pass_yards_per_point) + (show_proj['Pass TD']*pts_passtd) + (show_proj['Int']*pts_passint) + (show_proj['Rush Yards']/pts_rushrecyd) + (show_proj['Rec Yards']/pts_rushrecyd) + (show_proj['Rec']*pts_rec) + (show_proj['Rush TD']*pts_rushrectd) + (show_proj['Rec TD']*pts_rushrectd)
            show_proj['Proj FPts'] = round(show_proj['Proj FPts'],0)
            show_proj['Pass Yards'] = round(show_proj['Pass Yards'],0)
            show_proj['Rush Yards'] = round(show_proj['Rush Yards'],0)
            show_proj['Rec Yards'] = round(show_proj['Rec Yards'],0)
            show_proj['Pass Att'] = round(show_proj['Pass Att'],0)
            show_proj['Rush Att'] = round(show_proj['Rush Att'],0)
            show_proj['Rec'] = round(show_proj['Rec'],0)
            show_proj['Pass TD'] = round(show_proj['Pass TD'],1)
            show_proj['Rec TD'] = round(show_proj['Rec TD'],1)
            show_proj['Rush TD'] = round(show_proj['Rush TD'],1)
            show_proj['Int'] = round(show_proj['Int'],1)

            show_proj = show_proj.sort_values(by='Proj FPts',ascending=False)
            st.dataframe(show_proj[show_cols], hide_index=True, height=570, width=950)

            # Text input for custom file name with a default value
            c_col1, c_col2, c_col3 = st.columns([1,2,3])
            with c_col1:
                file_name = 'data_extract.csv'#st.text_input("Download Data as CSV:", value="projdata.csv")

                # Convert DataFrame to CSV
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv = convert_df_to_csv(show_proj)
                st.download_button(label="Download CSV", data=csv, file_name=file_name, mime='text/csv')
