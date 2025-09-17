import streamlit as st
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Define the correct password (replace with your desired password)
#st.markdown("<h1>Enter Password to Access Slate Analysis Tool",unsafe_allow_html=True)
CORRECT_PASSWORD = "foster"
CORRECT_PASSWORD2 = '1'

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Function to check password
def check_password():
    def password_entered():
        if (st.session_state["password"] == CORRECT_PASSWORD) or (st.session_state["password"] == CORRECT_PASSWORD2):
            st.session_state.authenticated = True
            del st.session_state["password"]  # Clear password from session state
        else:
            st.error("Incorrect password. Please try again.")
    
    if not st.session_state.authenticated:
        st.text_input("Enter Password (can be found in Resource Glossary at ftmff.substack.com", type="password", key="password", on_change=password_entered)
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
        name_change = pd.read_csv(f'{file_path}/nflnamechange.csv')
        allproplines = pd.read_csv(f'{file_path}/AllPropsData.csv')
        weekproj = pd.read_csv(f'{file_path}/ja_proj.csv')
        schedule = pd.read_csv(f'{file_path}/nfl_schedule_tracking.csv')
        dkdata = pd.read_csv(f'{file_path}/DKData.csv')
        implied_totals = pd.read_csv(f'{file_path}/implied_totals.csv')
        nfl_week_maps = pd.read_csv(f'{file_path}/nfl_week_mapping.csv')
        team_name_change = pd.read_csv(f'{file_path}/nflteamnamechange.csv')
        saltrack = pd.read_csv(f'{file_path}/DKSalTracking.csv')
        saltrack2 = pd.read_csv(f'{file_path}/DKSalTracking2.csv')
        xfp = pd.read_csv(f'{file_path}/xFPData.csv')

        return xfp, logo, adp_data, season_proj, name_change, allproplines, weekproj, schedule, dkdata, implied_totals, nfl_week_maps, team_name_change, saltrack,saltrack2
        
    def load_team_logos_dumb():
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, 'Data', 'logos')
        #ari = pd.read_csv(f'{file_path}/ari.png')
        with open(f'{file_path}/ari.png', 'rb') as file: ari = file.read()
        with open(f'{file_path}/atl.png', 'rb') as file: atl = file.read()
        with open(f'{file_path}/bal.png', 'rb') as file: bal = file.read()
        with open(f'{file_path}/buf.png', 'rb') as file: buf = file.read()

        with open(f'{file_path}/car.png', 'rb') as file: car = file.read()
        with open(f'{file_path}/chi.png', 'rb') as file: chi = file.read()
        with open(f'{file_path}/cin.png', 'rb') as file: cin = file.read()
        with open(f'{file_path}/cle.png', 'rb') as file: cle = file.read()

        with open(f'{file_path}/dal.png', 'rb') as file: dal = file.read()
        with open(f'{file_path}/den.png', 'rb') as file: den = file.read()
        with open(f'{file_path}/det.png', 'rb') as file: det = file.read()
        with open(f'{file_path}/gnb.png', 'rb') as file: gnb = file.read()

        with open(f'{file_path}/hou.png', 'rb') as file: hou = file.read()
        with open(f'{file_path}/ind.png', 'rb') as file: ind = file.read()
        with open(f'{file_path}/jax.png', 'rb') as file: jax = file.read()
        with open(f'{file_path}/ind.png', 'rb') as file: ind = file.read()
        
        with open(f'{file_path}/kan.png', 'rb') as file: kan = file.read()
        with open(f'{file_path}/lac.png', 'rb') as file: lac = file.read()
        with open(f'{file_path}/lar.png', 'rb') as file: lar = file.read()
        with open(f'{file_path}/lvr.png', 'rb') as file: lvr = file.read()

        with open(f'{file_path}/mia.png', 'rb') as file: mia = file.read()
        with open(f'{file_path}/min.png', 'rb') as file: min = file.read()
        with open(f'{file_path}/nor.png', 'rb') as file: nor = file.read()
        with open(f'{file_path}/nwe.png', 'rb') as file: nwe = file.read()

        with open(f'{file_path}/nyg.png', 'rb') as file: nyg = file.read()
        with open(f'{file_path}/nyj.png', 'rb') as file: nyj = file.read()
        with open(f'{file_path}/phi.png', 'rb') as file: phi = file.read()
        with open(f'{file_path}/pit.png', 'rb') as file: pit = file.read()

        with open(f'{file_path}/sea.png', 'rb') as file: sea = file.read()
        with open(f'{file_path}/sfo.png', 'rb') as file: sfo = file.read()
        with open(f'{file_path}/tam.png', 'rb') as file: tam = file.read()
        with open(f'{file_path}/ten.png', 'rb') as file: ten = file.read()
        with open(f'{file_path}/was.png', 'rb') as file: was = file.read()

        return ari,atl,bal,buf,car,chi,cin,cle,dal,den,det,gnb,hou,ind,jax,kan,lac,lar,lvr,mia,min,nor,nwe,nyg,nyj,phi,pit,sea,sfo,tam,ten,was
    
    def load_team_logos():
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, 'Data', 'logos')
        
        # Dictionary to store team logos
        team_logos = {}
        
        # List of team abbreviations
        teams = ['ari', 'atl', 'bal', 'buf', 'car', 'chi', 'cin', 'cle', 'dal', 'den', 'det', 
                'gnb', 'hou', 'ind', 'jax', 'kan', 'lac', 'lar', 'lvr', 'mia', 'min', 'nor', 
                'nwe', 'nyg', 'nyj', 'phi', 'pit', 'sea', 'sfo', 'tam', 'ten', 'was']
        
        # Load each team logo into the dictionary
        for team in teams:
            with open(f'{file_path}/{team}.png', 'rb') as file:
                team_logos[team] = file.read()
        
        return team_logos

    # Load the logos into a dictionary
    #team_logos = load_team_logos()
    
    #ari,atl,bal,buf,car,chi,cin,cle,dal,den,det,gnb,hou,ind,jax,kan,lac,lar,lvr,mia,min,nor,nwe,nyg,nyj,phi,pit,sea,sfo,tam,ten,was = load_team_logos()

    xfp, logo, adp_data, season_proj, namemap, allproplines, weekproj, schedule, dkdata, implied_totals, nfl_week_maps, team_name_change, saltrack,saltrack2 = load_data()

    dkdata['Rand'] = np.random.uniform(low=0.85, high=1.15, size=len(dkdata))
    dkdata['Own'] = round(dkdata['Own'] * dkdata['Rand'],0)

    own_dict = dict(zip(dkdata.Player,dkdata.Own))

    teamnamechangedict = dict(zip(team_name_change.Long,team_name_change.Short))
    check_matchups_dk = dict(zip(dkdata.Team,dkdata.Opp))
    check_matchups_proj = dict(zip(weekproj.Team,weekproj.Opp))

    this_week_number = dkdata['Week'].iloc[0]

    check_team = dkdata['Team'].iloc[0]
    check_team_dk = check_matchups_dk.get(check_team)
    check_team_proj = check_matchups_proj.get(check_team)

    if check_team_dk == check_team_proj:
        proj_are_good = 'Y'
    else:
        proj_are_good = 'N'
   
    all_game_times = schedule[['ID','Date','Time']]
    all_game_times['Date'] = pd.to_datetime(all_game_times['Date'])
    all_game_times['Date'] = all_game_times['Date'].dt.date
    all_game_times['DOW'] = all_game_times['Date'].apply(lambda x: x.strftime('%A'))
    all_game_times['start_hour'] = all_game_times['Time'].str.split(':').str[0].astype(int)
    all_game_times['MainSlate'] = np.where((all_game_times['DOW']=='Sunday')&(all_game_times['start_hour']>=13)&(all_game_times['start_hour']<17),'Y','N')
    main_slate_dict = dict(zip(all_game_times.ID,all_game_times.MainSlate))

    # this week
    get_this_week_number = dkdata['Week'].iloc[0]

    nfl_week_maps['Date'] = pd.to_datetime(nfl_week_maps['Date'])
    nfl_week_maps['Date'] = nfl_week_maps['Date'].dt.date
    schedule['Date'] = pd.to_datetime(schedule['Date'])
    schedule['Date'] = schedule['Date'].dt.date
    schedule = pd.merge(schedule,nfl_week_maps, on='Date',how='left')
    this_week_schedule = schedule[schedule['Week']==get_this_week_number]
    this_week_schedule['MainSlate'] = this_week_schedule['ID'].map(main_slate_dict)
    this_week_schedule['AwayShort'] = this_week_schedule['Away'].replace(teamnamechangedict)
    this_week_schedule['HomeShort'] = this_week_schedule['Home'].replace(teamnamechangedict)
    this_week_mainslate = this_week_schedule[this_week_schedule['MainSlate']=='Y'].reset_index(drop=True)
    this_week_mainslate = this_week_mainslate[['Away','Home']].drop_duplicates()
    this_week_mainslate['AwayShort'] = this_week_mainslate['Away'].replace(teamnamechangedict)
    this_week_mainslate['HomeShort'] = this_week_mainslate['Home'].replace(teamnamechangedict)
    away_list = list(this_week_mainslate['AwayShort'])
    home_list = list(this_week_mainslate['HomeShort'])
    main_slate_team_list = []
    for team in away_list:
        main_slate_team_list.append(team)
    for team in home_list:
        main_slate_team_list.append(team)
    
    ###
    weekproj['Team'] = weekproj['Team'].replace({'GB':'GNB','TB':'TAM','ARZ':'ARI','SF':'SFO'})
    weekproj['Opp'] = weekproj['Opp'].replace({'GB':'GNB','TB':'TAM', 'ARZ':'ARI','SF':'SFO'})
    season_proj['Proj FPts'] = 0
    namemapdict = dict(zip(namemap.OldName,namemap.NewName))
    adp_data['Player'] = adp_data['Player'].replace(namemapdict)
    season_proj['Player'] = season_proj['Player'].replace(namemapdict)
    implied_totals['Rank'] = implied_totals['Implied'].rank(ascending=False)
    
    teamnamechangedict = dict(zip(team_name_change.Long,team_name_change.Short))

    dkdata['Sal'] = pd.to_numeric(dkdata['Sal'])
    dkdata['Sal'] = dkdata['Sal'].astype(int)

    # get current adp
    adp_data = adp_data.sort_values(by='Date')
    last_ten_dates = adp_data['Date'].unique()[-7:]
    last_ten_adp = adp_data[adp_data['Date'].isin(last_ten_dates)].groupby('Player',as_index=False)['ADP'].mean().sort_values(by='ADP')
    last_ten_adp = last_ten_adp.round(1)
    curr_adp_dict = dict(zip(last_ten_adp.Player,last_ten_adp.ADP))
    curr_trend_dict = dict(zip(adp_data.Player,adp_data.Trend))
    
    st.sidebar.image(logo, width=250)  # Added logo to sidebar
    st.sidebar.title("Fantasy Football Resources")
    tab = st.sidebar.radio("Select View", ["Weekly Projections","Game by Game", "Season Projections","Salary Tracking", "Expected Fantasy Points", "Props","ADP Data","Tableau"], help="Choose a Page")
    
    if "reload" not in st.session_state:
        st.session_state.reload = False

    if st.sidebar.button("Reload Data"):
        st.session_state.reload = True
        st.cache_data.clear()  # Clear cache to force reload

    # Main content
    st.markdown(f"<center><h1>Follow The Money Fantasy Football Web App</h1></center>", unsafe_allow_html=True)

    if tab == "Expected Fantasy Points":
        st.markdown("<h1><center>Expected Fantasy Points Model</h1></center>", unsafe_allow_html=True)
        st.markdown("<i><center><h3>work in progress...</i></h3></center>", unsafe_allow_html=True)
        xpc1,xpc2,xpc3 = st.columns([1,1,1])
        with xpc1:
            selected_team = st.selectbox("Select Team", ["All"] + sorted(xfp['Team'].unique().tolist()))
        with xpc2:
            selected_pos = st.selectbox("Select Position", ["All"] + sorted(xfp['Pos'].unique().tolist()))
        with xpc3:
            selected_player = st.selectbox("Select Player", ["All"] + sorted(xfp['Player'].unique().tolist()))

        # Week slider for custom range
        #week_range = st.slider("Select Week Range", min_value=int(xfp['Week'].min()), max_value=int(xfp['Week'].max()),value=(int(xfp['Week'].min()), int(xfp['Week'].max())))

        # Filter the dataframe based on selections
        filtered_xfp = xfp.copy()
        if selected_team != "All":
            filtered_xfp = filtered_xfp[filtered_xfp['Team'] == selected_team]
        if selected_pos != "All":
            filtered_xfp = filtered_xfp[filtered_xfp['Pos'] == selected_pos]
        if selected_player != "All":
            filtered_xfp = filtered_xfp[filtered_xfp['Player'] == selected_player]
        #filtered_xfp = filtered_xfp[(filtered_xfp['Week'] >= week_range[0]) & (filtered_xfp['Week'] <= week_range[1])]

        # Display the filtered table
        st.dataframe(filtered_xfp[['Player', 'Pos', 'Team', 'Week', 'Passing', 'Rushing', 'Receiving', 'xFP']],hide_index=True)
        


    if tab == "Game by Game":
        st.markdown("<h1><center>Game by Game Preview</h1></center>", unsafe_allow_html=True)

        dksalsdf = dkdata[['Player','Sal']]

        show_all_game_info = st.checkbox('Show Game Info', value=False)
        if show_all_game_info:
            show_schedule = implied_totals[['Team','Opp','OU','Spread','Implied']].sort_values(by='OU',ascending=False)
            show_schedule['MainSlate'] = np.where(show_schedule['Team'].isin(main_slate_team_list),"Y","N")
            scol1, scol2, scol3 = st.columns([1,1,1])
            with scol2:
                st.dataframe(show_schedule, width=800,height=900, hide_index=True)

        get_this_week_number = dkdata['Week'].iloc[0]
        try:
            schedule=schedule.drop(['Week'],axis=1)
        except:
            pass
        schedule['Date'] = pd.to_datetime(schedule['Date'])
        schedule['Date'] = schedule['Date'].dt.date
        nfl_week_maps['Date'] = pd.to_datetime(nfl_week_maps['Date'])
        nfl_week_maps['Date'] = nfl_week_maps['Date'].dt.date
        schedule = pd.merge(schedule,nfl_week_maps, on='Date',how='left')

        this_week = schedule[schedule['Week']==get_this_week_number]
        this_week['Game Name'] =this_week['Away'] + ' @ ' + this_week['Home']
        game_selection_list = list(this_week['Game Name'].unique())
        game_selection = st.selectbox('Select A Game', game_selection_list)

        selectedgamedata = this_week[this_week['Game Name']==game_selection]
        game_date = selectedgamedata['Date'].iloc[0]
        game_time = selectedgamedata['Time'].iloc[0]
        date_to_show = game_date.strftime('%A, %B %-d')

        from datetime import datetime
        time_obj = datetime.strptime(game_time, '%H:%M')
        # Format to 12-hour time with AM/PM
        time_to_show = time_obj.strftime('%-I:%M %p')

        selected_gameid = selectedgamedata['ID'].iloc[0]
        game_line_log = schedule[schedule['ID']==selected_gameid]

        selectedgamedata = selectedgamedata[selectedgamedata['Timestamp']==np.max(selectedgamedata['Timestamp'])]

        road_team = selectedgamedata['Away'].iloc[0]
        home_team = selectedgamedata['Home'].iloc[0]

        road_team_short = teamnamechangedict.get(road_team)
        road_team_short_lower = road_team_short.lower()
        home_team_short = teamnamechangedict.get(home_team)
        home_team_short_lower = home_team_short.lower()

        favored_team = selectedgamedata['Underdog'].iloc[0]
        road_spread = selectedgamedata['Away Spread'].iloc[0]
        home_spread = selectedgamedata['Home Spread'].iloc[0]
        game_ou = selectedgamedata['OU'].iloc[0]

        if favored_team == road_team:
            favored_spread = road_spread
        else:
            favored_spread = home_spread

        #st.markdown(f"<h1><center>{road_team} vs. {home_team}</h1></center>", unsafe_allow_html=True)
        #st.markdown(f"<h3><center>{favored_team} {favored_spread} <br> Over/Under: {game_ou}</h3></center>",unsafe_allow_html=True)

        st.markdown(f"""<center><font size=25 face=Futura><b>{road_team} vs. {home_team}</b></font><br>
                        <center><font size=6 face=Futura><u>{date_to_show} at {time_to_show}</u></center></font>
                        <font size=6 face=Futura><i><b>{favored_team} {favored_spread}</font><br>
                        <font size=6 face=Futura>Over/Under: {game_ou}</center></font><br>
                    """, unsafe_allow_html=True)
        
        line_move_check = st.checkbox('Show Line Movements', value=False)
        if line_move_check:
            ndcol1,ndcol2 = st.columns([1,6])
            with ndcol1:
                numdays = st.number_input('Number of days back',3,10,value=7)
                game_line_log = game_line_log.tail(numdays)

            # Create smaller line plots with reduced text sizes
            plotcol1, plotcol2 = st.columns([1, 1])
            min_ou = np.min(game_line_log['OU'])
            max_ou = np.max(game_line_log['OU'])
            min_spread = np.min(game_line_log['Home Spread'])
            max_spread = np.max(game_line_log['Home Spread'])
            with plotcol1:
                fig, ax = plt.subplots(figsize=(2.4, 1.5))  # Smaller figure size
                game_line_log.plot.line(x='Timestamp', y='OU', ax=ax, linewidth=1)
                ax.get_legend().remove()  # Hide the legend
                ax.set_xlabel("Timestamp", fontsize=4)
                ax.set_ylim([min_ou-2,max_ou+2])
                ax.set_ylabel("OU", fontsize=4)
                ax.set_title("OU Over Time", fontsize=6)
                ax.tick_params(axis='both', labelsize=3)  # Smaller tick labels
                plt.xticks(rotation=45)
                plt.tight_layout()  # Adjust layout to prevent clipping
                st.pyplot(fig)

            with plotcol2:
                fig, ax = plt.subplots(figsize=(2.4, 1.5))  # Smaller figure size
                game_line_log.plot.line(x='Timestamp', y='Home Spread', ax=ax, linewidth=1)
                ax.get_legend().remove()  # Hide the legend
                ax.set_xlabel("Timestamp", fontsize=4)
                ax.set_ylim([min_spread-1,max_spread+1])
                ax.set_ylabel("Spread", fontsize=4)  # Corrected label to match data
                ax.set_title("Spread Over Time", fontsize=6)  # Corrected title
                ax.tick_params(axis='both', labelsize=3)  # Smaller tick labels
                plt.xticks(rotation=45)
                plt.tight_layout()  # Adjust layout to prevent clipping
                st.pyplot(fig)

        weekproj = pd.merge(weekproj,dksalsdf,how='left',on='Player')
        road_projections = weekproj[weekproj['Team']==road_team_short]
        road_implied = implied_totals[implied_totals['Team']==road_team_short]['Implied'].iloc[0]
        road_implied_rank = implied_totals[implied_totals['Team']==road_team_short]['Rank'].iloc[0]
        home_projections = weekproj[weekproj['Team']==home_team_short]
        home_implied = implied_totals[implied_totals['Team']==home_team_short]['Implied'].iloc[0]
        home_implied_rank = implied_totals[implied_totals['Team']==home_team_short]['Rank'].iloc[0]
        
        if proj_are_good == 'N':
            st.markdown(f'<h2><center>Projections for week {this_week_number} are not yet available</center></h2>',unsafe_allow_html=True)
            pass
        else:

            projcol1, projcol2 = st.columns([1,1])

            with projcol1:
                st.markdown(f"<center><font size=13><b>{road_team}</b></font><br><font size=4><i>Implied for {road_implied} points, ranked #{int(road_implied_rank)} of {len(implied_totals)}</i></center><hr>", unsafe_allow_html=True)
                #st.image(team_logos.get(road_team_short_lower),width=200)
                st.markdown("<h4>Quarterback</h4>",unsafe_allow_html=True)
                road_qb_proj = road_projections[road_projections['Pos']=='QB'][['Player','Sal','Pass Comp','Pass Att','Pass Yards','Pass TD', 'Int','Rush Att','Rush Yds','Rush TD']].sort_values(by='Pass Att',ascending=False)
                st.dataframe(road_qb_proj, hide_index=True, width=630)
                st.markdown("<h4>Running Backs</h4>",unsafe_allow_html=True)
                road_rb_proj = road_projections[road_projections['Pos']=='RB'][['Player','Sal','Rush Att','Rush Yds','Rush TD','Tgt','Rec','Rec Yds','Rec TD']].sort_values(by='Rush Att',ascending=False)
                st.dataframe(road_rb_proj, hide_index=True, width=600,height=150)
                st.markdown("<h4>Pass Catchers</h4>",unsafe_allow_html=True)
                road_rec_proj = road_projections[road_projections['Pos'].isin(['WR','TE'])][['Player','Sal','Tgt','Rec','Rec Yds','Rec TD']].sort_values(by='Rec Yds',ascending=False)
                if len(road_rec_proj) > 7:
                    st.dataframe(road_rec_proj, hide_index=True, width=600,height=325)
                else:
                    st.dataframe(road_rec_proj, hide_index=True, width=600)

            with projcol2:
                st.markdown(f"<center><font size=13><b>{home_team}</b></font><br><font size=4><i>Implied for {home_implied} points, ranked #{int(home_implied_rank)} of {len(implied_totals)}</i></center><hr>", unsafe_allow_html=True)
                st.markdown("<h4>Quarterback</h4>",unsafe_allow_html=True)
                home_qb_proj = home_projections[home_projections['Pos']=='QB'][['Player','Sal','Pass Comp','Pass Att','Pass Yards','Pass TD', 'Int','Rush Att','Rush Yds','Rush TD']].sort_values(by='Pass Att',ascending=False)
                st.dataframe(home_qb_proj, hide_index=True, width=630)
                st.markdown("<h4>Running Backs</h4>",unsafe_allow_html=True)
                home_rb_proj = home_projections[home_projections['Pos']=='RB'][['Player','Sal','Rush Att','Rush Yds','Rush TD','Tgt','Rec','Rec Yds','Rec TD']].sort_values(by='Rush Att',ascending=False)
                st.dataframe(home_rb_proj, hide_index=True, width=600,height=150)
                st.markdown("<h4>Pass Catchers</h4>",unsafe_allow_html=True)
                home_rec_proj = home_projections[home_projections['Pos'].isin(['WR','TE'])][['Player','Sal','Tgt','Rec','Rec Yds','Rec TD']].sort_values(by='Rec Yds',ascending=False)
                if len(home_rec_proj) > 7:
                    st.dataframe(home_rec_proj, hide_index=True, width=600,height=325)
                else:
                    st.dataframe(home_rec_proj, hide_index=True, width=600)

    if tab == "Weekly Projections":
        st.markdown("<h3><center>Weekly Projections & Ranks</h3></center>", unsafe_allow_html=True)

        if proj_are_good == 'N':
            st.markdown(f'<h2><center>Projections for week {this_week_number} are not yet available</center></h2>',unsafe_allow_html=True)
            pass
        else:

            # Initialize session state for scoring settings
            if 'scoring_settings' not in st.session_state:
                st.session_state.scoring_settings = {
                    'pass_yards': 25.0,
                    'pass_td': 4.0,
                    'interception': -1.0,
                    'rush_yards': 10.0,
                    'rush_td': 6.0,
                    'rec_yards': 10.0,
                    'reception': 1.0,
                    'rec_td': 6.0
                }

            # Position filter
            weekprojcol1, weekprojcol2, weekprojcol3 = st.columns([1,1,1])
            with weekprojcol1:
                positions = ['All', 'QB', 'RB', 'WR', 'TE', 'FLEX']
                selected_position = st.selectbox("Select Position", positions)
            with weekprojcol2:
                # Team filter
                teams = ['All'] + sorted(weekproj['Team'].unique().tolist())
                selected_team = st.selectbox("Select Team", teams)
            with weekprojcol3:
                mainslateselect = st.selectbox('Show Main Slate Only', ['No','Yes'])
            
            # Button to toggle scoring settings
            show_scoring = st.button("Customize Scoring System", key="toggle_scoring")

            # Show scoring settings if button is clicked
            if show_scoring or st.session_state.get('show_settings', False):
                st.session_state.show_settings = True
                with st.expander("Scoring Settings", expanded=True):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.session_state.scoring_settings['pass_yards'] = st.number_input(
                            "Pass Yards per Point", 
                            value=st.session_state.scoring_settings['pass_yards'], 
                            step=0.1, 
                            key="pass_yards"
                        )
                        st.session_state.scoring_settings['pass_td'] = st.number_input(
                            "Pass TD Points", 
                            value=st.session_state.scoring_settings['pass_td'], 
                            step=0.1, 
                            key="pass_td"
                        )
                        st.session_state.scoring_settings['interception'] = st.number_input(
                            "Interception Points", 
                            value=st.session_state.scoring_settings['interception'], 
                            step=0.1, 
                            key="interception"
                        )

                    with col2:
                        st.session_state.scoring_settings['rush_yards'] = st.number_input(
                            "Rush Yards per Point", 
                            value=st.session_state.scoring_settings['rush_yards'], 
                            step=0.1, 
                            key="rush_yards"
                        )
                        st.session_state.scoring_settings['rush_td'] = st.number_input(
                            "Rush TD Points", 
                            value=st.session_state.scoring_settings['rush_td'], 
                            step=0.1, 
                            key="rush_td"
                        )

                    with col3:
                        st.session_state.scoring_settings['rec_yards'] = st.number_input(
                            "Receiving Yards per Point", 
                            value=st.session_state.scoring_settings['rec_yards'], 
                            step=0.1, 
                            key="rec_yards"
                        )
                        st.session_state.scoring_settings['reception'] = st.number_input(
                            "Reception Points", 
                            value=st.session_state.scoring_settings['reception'], 
                            step=0.1, 
                            key="reception"
                        )
                        st.session_state.scoring_settings['rec_td'] = st.number_input(
                            "Receiving TD Points", 
                            value=st.session_state.scoring_settings['rec_td'], 
                            step=0.1, 
                            key="rec_td"
                        )

            # Calculate fantasy points based on scoring settings
            weekproj['FPts'] = (
                (weekproj['Pass Yards'] / st.session_state.scoring_settings['pass_yards']) +
                (weekproj['Pass TD'] * st.session_state.scoring_settings['pass_td']) +
                (weekproj['Int'] * st.session_state.scoring_settings['interception']) +
                (weekproj['Rush Yds'] / st.session_state.scoring_settings['rush_yards']) +
                (weekproj['Rec Yds'] / st.session_state.scoring_settings['rec_yards']) +
                (weekproj['Rec'] * st.session_state.scoring_settings['reception']) +
                (weekproj['Rush TD'] * st.session_state.scoring_settings['rush_td']) +
                (weekproj['Rec TD'] * st.session_state.scoring_settings['rec_td'])
            )

            weekproj['Own'] = weekproj['Player'].map(own_dict)

            # ranking
            if mainslateselect == 'Yes':
                weekproj = weekproj[weekproj['Team'].isin(main_slate_team_list)]
            weekproj['Pos Rank'] = weekproj.groupby('Pos')['FPts'].rank(ascending=False)

            dfs_sals_check = st.checkbox('Show DFS Info?')
            dfs_sals_dict = dict(zip(dkdata.Player,dkdata.Sal))

            # Filter data based on selections
            filtered_data = weekproj.copy()
            
            if selected_position != 'All':
                if selected_position == 'FLEX':
                    filtered_data = filtered_data[filtered_data['Pos'].isin(['RB', 'WR', 'TE'])]
                else:
                    filtered_data = filtered_data[filtered_data['Pos'] == selected_position]
            
            if selected_team != 'All':
                filtered_data = filtered_data[filtered_data['Team'] == selected_team]

            # Round and sort
            filtered_data = filtered_data.round(2)
            filtered_data = filtered_data.sort_values(by='FPts', ascending=False)
            filtered_data['Sal'] = filtered_data['Player'].map(dfs_sals_dict)
            filtered_data['Sal'] = filtered_data['Sal'].fillna(0)

            mask = pd.to_numeric(filtered_data['Sal'], errors='coerce').isna()
            rows_with_strings = filtered_data[mask]

            # Replace non-numeric values in 'Sal' with 0
            filtered_data.loc[mask, 'Sal'] = 0

            filtered_data['Sal'] = pd.to_numeric(filtered_data['Sal'])

            # Now the calculation should work
            filtered_data['Val'] = filtered_data['FPts'] / (filtered_data['Sal'] / 1000)
            filtered_data['Val'] = round(filtered_data['Val'],2)

            filtered_data = filtered_data[filtered_data['FPts']>0]

            #filtered_data['Sal'] = filtered_data['Sal'].apply(lambda x: f"${int(x):,}")
            #filtered_data['Val'] = round(filtered_data['FPts']/(filtered_data['Sal']/1000),2)

            # display based on position selection
            if selected_position == 'All':
                if dfs_sals_check:
                    proj_show_cols = ['Player','Team','Opp','Sal','Own','FPts','Val','Pos Rank','Pass Comp','Pass Att','Pass Yards','Pass TD','Int','Rush Att','Rush Yds','Rush TD','Tgt','Rec','Rec Yds','Rec TD']
                else:
                    proj_show_cols = ['Player','Team','Opp','FPts','Pos Rank','Pass Comp','Pass Att','Pass Yards','Pass TD','Int','Rush Att','Rush Yds','Rush TD','Tgt','Rec','Rec Yds','Rec TD']
            elif selected_position == 'QB':
                if dfs_sals_check:
                    proj_show_cols = ['Player','Team','Opp','Sal','Own','FPts','Val','Pos Rank','Pass Comp','Pass Att','Pass Yards','Pass TD','Int','Rush Att','Rush Yds','Rush TD']
                else:
                    proj_show_cols = ['Player','Team','Opp','FPts','Pos Rank','Pass Comp','Pass Att','Pass Yards','Pass TD','Int','Rush Att','Rush Yds','Rush TD']
            elif selected_position in ['WR','TE']:
                if dfs_sals_check:
                    proj_show_cols = ['Player','Team','Opp','Sal','Own','FPts','Val','Pos Rank','Tgt','Rec','Rec Yds','Rec TD']
                else:
                    proj_show_cols = ['Player','Team','Opp','FPts','Pos Rank','Tgt','Rec','Rec Yds','Rec TD']
            elif selected_position in ['RB','WR','TE','FLEX']:
                if dfs_sals_check:
                    proj_show_cols = ['Player','Team','Opp','Sal','Own','FPts','Val','Pos Rank','Rush Att','Rush Yds','Rush TD','Tgt','Rec','Rec Yds','Rec TD']
                else:
                    proj_show_cols = ['Player','Team','Opp','FPts','Pos Rank','Rush Att','Rush Yds','Rush TD','Tgt','Rec','Rec Yds','Rec TD']

            # Display the filtered dataframe
            show_filtered_data = filtered_data[proj_show_cols]
            st.dataframe(show_filtered_data, hide_index=True, height=750)
            csv = convert_df_to_csv(show_filtered_data)
            st.download_button(label="Download CSV", data=csv, file_name='JA Projections.csv', mime='text/csv')

    if tab == "Salary Tracking":
        st.markdown("<h1><center>DraftKings Salary Tracking</h1></center>", unsafe_allow_html=True)
        saltrack = saltrack.drop(['Unnamed: 0'],axis=1)
        saltrack2 = saltrack2.drop(['Unnamed: 0'],axis=1)

        #st.markdown("<h3><center>Full Table</h3></center>", unsafe_allow_html=True)
        player_name_list = list(saltrack['Player'])
        b1,b2,b3 = st.columns([1,1,1])
        with b2:
            sal_select = st.selectbox('Select Player', ['All'] + player_name_list)
        if sal_select == 'All':
            st.dataframe(saltrack, hide_index=True, height=1000)
        else:
            psaldf = saltrack[saltrack['Player']==sal_select]
            psaldf2 = saltrack2[saltrack2['Player']==sal_select]
            minsal = np.min(psaldf2['Sal'])
            maxsal = np.max(psaldf2['Sal'])
            p_weeklist = list(psaldf2['Week'])
            st.dataframe(psaldf, hide_index=True)

            sal_plot_col1, sal_plot_col2, sal_plot_col3 = st.columns([1,2,1])
            with sal_plot_col2:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(psaldf2['Week'], psaldf2['Sal'], color='red', linewidth=3, marker='o')  # Thicker red line
                ax.set_xlabel('Week')
                ax.set_ylabel('Salary')
                ax.set_title(f'{sal_select} Price by Week')
                ax.set_ylim(minsal-500, maxsal+500)
                ax.grid(True)
                ax.set_xticks(p_weeklist)

                # Add annotations for salary values
                for i, row in psaldf2.iterrows():
                    ax.annotate(row['Sal'], (row['Week'], row['Sal']), textcoords="offset points", xytext=(0,10), ha='center')

                # Display the plot in Streamlit
                st.pyplot(fig)

                        

            
        




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
        st.markdown(f"<center><h2>Season Long Projections</h2></center>", unsafe_allow_html=True)

        a_col1, a_col2 = st.columns([1,4])
        with a_col1:
            box_col1, box_col2 = st.columns([1,1])
            with box_col1:
                show_qb = st.checkbox('QB', value=True)
                show_rb = st.checkbox('RB', value=True)
            with box_col2:
                show_wr = st.checkbox('WR', value=True)
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
                show_cols = ['Player','Team','Pos','ADP','ADP Trend','Proj FPts','Rush Att','Rush Yards','Rush TD','Rec','Rec Yards','Rec TD']
            elif ('QB' in pos_selected) & ('RB' not in pos_selected) & ('WR' not in pos_selected) & ('TE' not in pos_selected):
                show_cols = ['Player','Team','Pos','ADP','ADP Trend','Proj FPts','Pass Att','Pass Yards','Int','Pass TD','Rush Att','Rush Yards','Rush TD']
            else:
                show_cols = ['Player','Team','Pos','ADP','ADP Trend','Proj FPts','Pass Att','Pass Yards','Int','Pass TD','Rush Att','Rush Yards','Rush TD','Rec','Rec Yards','Rec TD']

            show_proj = season_proj[season_proj['Pos'].isin(pos_selected)]
            show_proj['ADP'] = show_proj['Player'].map(curr_adp_dict)
            show_proj['ADP Trend'] = show_proj['Player'].map(curr_trend_dict)

            projteamlist = list(season_proj['Team'].unique())
            projteamlist.sort()
            team_list = ['All'] + projteamlist 
            team_selection = st.selectbox("Team Filter:", team_list)
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

            show_proj = show_proj[show_proj['Proj FPts']>19].sort_values(by='Proj FPts',ascending=False)
            if team_selection == 'All':
                pass
            else:
                show_proj = show_proj[show_proj['Team']==team_selection]
            
            if len(show_proj) > 10:
                st.dataframe(show_proj[show_cols], hide_index=True, height=570, width=1200)
            else:
                st.dataframe(show_proj[show_cols], hide_index=True,  width=1200)

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
    
    elif tab == "Props":
        qb_markets = ['player_pass_completions','player_pass_attempts','player_pass_td','player_pass_yds']

        allproplines['game_key'] = allproplines['game_key'].str.replace('@',' @ ')
        game_list = list(allproplines['game_key'].unique())
        a_col1, a_col2 = st.columns([1,1])
        
        game_select = st.selectbox("Select A Game", game_list)
        game_lines = allproplines[allproplines['game_key']==game_select]
        col1,col2=st.columns([1,1])
        with col1:
            pos_select = st.radio('Select Position', ['QB','RB','WR/TE'])
        with col2:
            if pos_select == 'QB':
                select_game_data = game_lines[game_lines['market'].isin(qb_markets)]
                qb_options = list(select_game_data['player'].unique())
                qb_dropdown = st.selectbox('Select QB', qb_options)

        #st.dataframe(select_game_data)

        st.markdown("Prop Movements")
        player_list = list(allproplines['player'].unique())
        propplayerpick = st.selectbox('Select a Player', player_list)
        player_prop_lines = allproplines[allproplines['player']==propplayerpick]
        player_markets = list(player_prop_lines['market'].unique())
        player_books = list(player_prop_lines['book'].unique())
        marketpick = st.selectbox('Select a Market', player_markets)
        bookpick = st.selectbox('Select a Sports Book', player_books)

        selected_lines = allproplines[(allproplines['player']==propplayerpick)&(allproplines['market']==marketpick)&(allproplines['book']==bookpick)]
        selected_lines = selected_lines[selected_lines['over_under']=='Over']

        #st.dataframe(selected_lines)

        fig, ax = plt.subplots(figsize=(5, 3))  # Smaller figure size
        selected_lines.plot.line(x='scrape_time', y='line', ax=ax, linewidth=3)
        ax.get_legend().remove()  # Hide the legend
        ax.set_xlabel("Timestamp", fontsize=4)
        ax.set_ylabel("Line", fontsize=4)
        ax.set_title(f"{propplayerpick} {marketpick} Line Movement", fontsize=6)
        ax.tick_params(axis='both', labelsize=7)  # Smaller tick labels
        plt.xticks(rotation=45)
        plt.tight_layout()  # Adjust layout to prevent clipping
        st.pyplot(fig)



    
    elif tab == "Tableau":
        tableau_choice = st.selectbox(options=['ADP','NFL 2024'],label='Choose dashboard to display')
        if tableau_choice == 'ADP':
            #st.markdown("<h2><center>Main MLB Dashboard</center></h2>", unsafe_allow_html=True)
            st.markdown("<i><center><a href='https://public.tableau.com/app/profile/jon.anderson4212/viz/FTMFFADPDashboard/ADPTable#1'>Click here to visit full thing</i></a></center>", unsafe_allow_html=True)
            tableau_code_nfladp = """
            <div class='tableauPlaceholder' id='viz1755127779758' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;FT&#47;FTMFFADPDashboard&#47;ADPTable&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='FTMFFADPDashboard&#47;ADPTable' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;FT&#47;FTMFFADPDashboard&#47;ADPTable&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1755127779758');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1200px';vizElement.style.height='850px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1200px';vizElement.style.height='850px';} else { vizElement.style.width='100%';vizElement.style.height='800px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
            """ 
            components.html(tableau_code_nfladp, height=750, scrolling=True)
        elif tableau_choice == 'NFL 2024':
            #st.markdown("<h2><center>Main MLB Dashboard</center></h2>", unsafe_allow_html=True)
            st.markdown("<i><center><a href='https://public.tableau.com/app/profile/jon.anderson4212/viz/JonPGHNFL2024/TeamSummary'>Click here to visit full thing</i></a></center>", unsafe_allow_html=True)
            tableau_code_nfl24 = """
            <div class='tableauPlaceholder' id='viz1755127840102' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Jo&#47;JonPGHNFL2024&#47;TeamSummary&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='JonPGHNFL2024&#47;TeamSummary' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Jo&#47;JonPGHNFL2024&#47;TeamSummary&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1755127840102');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1300px';vizElement.style.height='850px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1300px';vizElement.style.height='850px';} else { vizElement.style.width='100%';vizElement.style.height='1500px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
            """ 
            components.html(tableau_code_nfl24, height=750, scrolling=True)