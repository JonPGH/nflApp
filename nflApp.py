import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import os
import warnings, io
warnings.filterwarnings("ignore")
import numpy as np
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import os, shutil, math

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
        mainslate = pd.read_csv(f'{file_path}/mainslate.csv')
        shootout_teams = pd.read_csv(f'{file_path}/shootout_team_data.csv')
        shootout_matchups = pd.read_csv(f'{file_path}/shootout_game_data.csv')
        bookproj = pd.read_csv(f'{file_path}/bookproj.csv')
        qb_grades = pd.read_csv(f'{file_path}/qb_grades.csv')
        rb_grades = pd.read_csv(f'{file_path}/rb_grades.csv')
        wr_grades = pd.read_csv(f'{file_path}/wr_grades.csv')
        te_grades = pd.read_csv(f'{file_path}/te_grades.csv')
        team_grades = pd.read_csv(f'{file_path}/team_grading.csv')
        optimizer_proj = pd.read_csv(f'{file_path}/main_slate_projections.csv')#

        return optimizer_proj,team_grades, qb_grades, rb_grades, wr_grades, te_grades, mainslate, shootout_teams, shootout_matchups, xfp, logo, adp_data, season_proj, name_change, allproplines, weekproj, schedule, dkdata, implied_totals, nfl_week_maps, team_name_change, saltrack,saltrack2,bookproj
        
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

    optimizer_proj,team_grades, qb_grades, rb_grades, wr_grades, te_grades, mainslate, shootout_teams, shootout_matchups, xfp, logo, adp_data, season_proj, namemap, allproplines, weekproj, schedule, dkdata, implied_totals, nfl_week_maps, team_name_change, saltrack,saltrack2,bookproj = load_data()
    mainslate['Rand'] = np.random.uniform(low=0.85, high=1.15, size=len(mainslate))
    mainslate['proj_own'] = round(mainslate['proj_own'] * mainslate['Rand'],0)

    qb_grades['Rk'] = qb_grades['Season'].rank(ascending=False).astype(int)
    qb_grade_dict = dict(zip(qb_grades.QB,qb_grades.Rk))
    
    rb_grades = rb_grades[rb_grades['Season']>1]
    rb_grades['Rk'] = rb_grades['Season'].rank(ascending=False).astype(int)
    rb_grade_dict = dict(zip(rb_grades.RB,rb_grades.Rk))
    
    wr_grades = wr_grades[wr_grades['Season']>1]
    wr_grades['Rk'] = wr_grades['Season'].rank(ascending=False).astype(int)
    wr_grade_dict = dict(zip(wr_grades.WR,wr_grades.Rk))
    
    te_grades = te_grades[te_grades['Season']>1]
    te_grades['Rk'] = te_grades['Season'].rank(ascending=False).astype(int)
    te_grade_dict = dict(zip(te_grades.TE,te_grades.Rk))

    all_grade_rank_dict = qb_grade_dict | rb_grade_dict | wr_grade_dict | te_grade_dict    
    
    own_dict = dict(zip(mainslate.name,mainslate.proj_own))

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

    shootout_matchups['GameSS'] = round(shootout_matchups['Game SS'],0)

    game_ss_dict = dict(zip(shootout_matchups.Team,shootout_matchups.GameSS))
    
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
    tab = st.sidebar.radio("Select View", ["Weekly Projections","Game by Game","DFS Optimizer","Book Based Proj","Player Grades","Salary Tracking", "Expected Fantasy Points", "Props","ADP Data","Tableau"], help="Choose a Page")
    
    if "reload" not in st.session_state:
        st.session_state.reload = False

    if st.sidebar.button("Reload Data"):
        st.session_state.reload = True
        st.cache_data.clear()  # Clear cache to force reload

    # Main content
    st.markdown(f"<center><h1>Follow The Money Fantasy Football Web App</h1></center>", unsafe_allow_html=True)

    def color_season(val):
        if pd.isna(val) or val == 'None' or not isinstance(val, (int, float, str)):
            return 'white'
        try:
            val = float(val)
            if val >= 100:
                r = int(255 * (1 - (val - 100) / 100))  # Green increases above 100
                g = 255
                b = int(255 * ((val - 100) / 100))
            else:
                r = 255
                g = int(255 * (val / 100))  # Red increases below 100
                b = int(255 * (val / 100))
                return f'rgb({r}, {g}, {b})'
        except (ValueError, TypeError):
            return 'white'

    if tab == "DFS Optimizer":
        st.markdown(f"""<br><center><font size=10 face=Futura><b>Follow The Money DFS Tool<br></b>
        <font size=3 face=Futura>These projections are tweaked slightly for more DFS friendly projections, including ceiling and positional adjustments.</font></center>""", unsafe_allow_html=True)
        
        show_projections_check = st.checkbox('Show Projections?', value=True)
        
        try:
            optimizer_proj = optimizer_proj.rename({'DK ID':'DKID'},axis=1)
        except:
            pass
        dk_id_dict = dict(zip(optimizer_proj.Player,optimizer_proj.DKID))

        account_for_ceiling_check = st.checkbox('Adjust for Ceiling', value=True)

        if account_for_ceiling_check:
            optimizer_proj = optimizer_proj[['Player', 'Pos','Team','Opp','Sal','wProj_ceil']]
            optimizer_proj.columns=['Player','Pos','Team','Opp','Sal','Proj']
            optimizer_proj['Value'] = optimizer_proj['Proj'] / (optimizer_proj['Sal']/1000)
        else:
            optimizer_proj = optimizer_proj[['Player', 'Pos','Team','Opp','Sal','wProj1']]
            optimizer_proj.columns=['Player','Pos','Team','Opp','Sal','Proj']
            optimizer_proj['Value'] = optimizer_proj['Proj'] / (optimizer_proj['Sal']/1000)

        # positional adjustment
        pos_value_medians = optimizer_proj.groupby('Pos',as_index=False)['Value'].median()
        pos_value_medians.columns=['Pos','PosV']
        optimizer_proj = pd.merge(optimizer_proj,pos_value_medians,on='Pos')
        optimizer_proj["Value"] = optimizer_proj.groupby("Pos")["Value"].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        optimizer_proj = optimizer_proj.drop(['PosV'],axis=1)
        optimizer_proj = optimizer_proj.round(2)

        # ------------------ FILTER CONTROLS ------------------
        # Three columns: Pos dropdown | Salary slider | Player search
        col1, col2, col3 = st.columns([1, 2, 2])

        # 1) Position dropdown (with "All")
        pos_options = ["All"] + sorted(optimizer_proj["Pos"].unique().tolist())
        pos_choice = col1.selectbox("Filter by Position:", pos_options, index=0)

        # Salary slider uses global min/max (int)
        sal_min = int(optimizer_proj["Sal"].min())
        sal_max = int(optimizer_proj["Sal"].max())
        salary_range = col2.slider(
            "Salary range ($)",
            min_value=sal_min,
            max_value=sal_max,
            value=(sal_min, sal_max),
            step=100
        )

        # 3) Player search
        player_search = col3.text_input("Search for Player:")

        # ------------------ APPLY FILTERS ------------------
        filtered = optimizer_proj.copy()

        # Position filter
        if pos_choice != "All":
            filtered = filtered[filtered["Pos"] == pos_choice]

        # Salary filter
        filtered = filtered[(filtered["Sal"] >= salary_range[0]) & (filtered["Sal"] <= salary_range[1])]

        # Player search (case-insensitive)
        if player_search:
            filtered = filtered[filtered["Player"].str.contains(player_search, case=False, na=False)]

        # Sort by Value desc
        filtered = filtered.sort_values(by='Value', ascending=False)

        # --- Row colors by position ---
        pos_colors = {
            "QB": "#e0f7fa",
            "RB": "#f1f8e9",
            "WR": "#fff3e0",
            "TE": "#fce4ec",
            "DST": "#ede7f6"
        }

        def highlight_rows(row):
            color = pos_colors.get(row["Pos"], "#ffffff")
            return [f"background-color: {color}"] * len(row)

        # --- Formatting: all numeric → 2 decimals; Sal → currency no decimals ---
        numeric_cols = filtered.select_dtypes(include="number").columns.tolist()
        fmt = {col: "{:.2f}" for col in numeric_cols}
        if "Sal" in fmt:
            fmt["Sal"] = "${:,.0f}"

        styler = (
            filtered.style
                .apply(highlight_rows, axis=1)
                .format(fmt)
        )

        showprojcol1, showprojcol2, showprojcol3 = st.columns([1,5,1])
        with showprojcol2:
            st.dataframe(styler, height=500, use_container_width=True, hide_index=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        ### OPTIMIZER CODE #################

        try:
            import pulp
        except Exception:
            pulp = None

        st.markdown("### 🧮 DFS Optimizer (DraftKings NFL)")

        # -------------------- SETTINGS UI --------------------
        with st.expander("Optimizer Settings", expanded=True):
            n_lineups = st.number_input("How many lineups to generate?", min_value=1, max_value=150, value=10, step=1)
            variance_pct = st.slider("Projection variance (±%) per lineup", 0, 50, 10, 1)
            max_exposure_pct = st.slider("Max player exposure (%)", 0, 100, 60, 5)
            dk_salary_cap = st.number_input("Salary Cap ($)", 30000, 70000, 50000, 500)

            # Exclude + Lock controls (from your version)
            all_player_opts = sorted(filtered["Player"].dropna().unique().tolist())
            exclude_players = st.multiselect("Exclude players (optional):", all_player_opts)
            lock_players    = st.multiselect("LOCK players (optional, forced into every lineup):", all_player_opts)

            # ---------- Pop-out Player Pool (include-only list) ----------
            base_ui_pool = filtered[(filtered["Sal"] > 0) & (filtered["Pos"].isin(["QB","RB","WR","TE","DST"]))].copy()
            pos_list = ["QB","RB","WR","TE","DST"]

            # Initialize per-position selections in session_state (defaults = Proj ≥ 5)
            for pos in pos_list:
                key = f"include_{pos}"
                pos_names = base_ui_pool.loc[base_ui_pool["Pos"]==pos, "Player"].dropna().tolist()
                default_names = base_ui_pool[(base_ui_pool["Pos"]==pos) & (base_ui_pool["Proj"]>=5)]["Player"].tolist()
                if key not in st.session_state:
                    st.session_state[key] = default_names
                else:
                    st.session_state[key] = [p for p in st.session_state[key] if p in pos_names]

            pop = st.popover("Player Pool (click to choose)")
            with pop:
                st.caption("Pick who to include in the build. Defaults to players with **Proj ≥ 5**.")
                tabs = st.tabs(pos_list + ["All selected"])
                for idx, pos in enumerate(pos_list):
                    with tabs[idx]:
                        pos_df = base_ui_pool[base_ui_pool["Pos"]==pos].sort_values("Proj", ascending=False)
                        options = pos_df["Player"].tolist()
                        labels = [f"{r.Player} — {r.Team}/{r.Opp}  ${int(r.Sal):,}  ({r.Proj:.2f})" for _, r in pos_df.iterrows()]
                        label_to_name = dict(zip(labels, options))
                        pre = [lbl for lbl in labels if label_to_name[lbl] in st.session_state[f"include_{pos}"]]
                        chosen_labels = st.multiselect(f"Include {pos}:", labels, default=pre, key=f"ms_{pos}")
                        st.session_state[f"include_{pos}"] = [label_to_name[lbl] for lbl in chosen_labels]
                with tabs[-1]:
                    chosen_all = sorted(set(sum([st.session_state[f"include_{p}"] for p in pos_list], [])))
                    st.write(f"**Selected players ({len(chosen_all)}):**")
                    st.write(", ".join(chosen_all) if chosen_all else "_None_")
                    if st.button("Clear all selections"):
                        for p in pos_list:
                            st.session_state[f"include_{p}"] = []

        # Primary action – ONLY runs the optimizer when pressed
        generate_clicked = st.button("🔁 Generate lineups", type="primary")

        # -------------------- SOLVER DISCOVERY --------------------
        def get_solver():
            if pulp is None:
                return None
            for p in ["/opt/homebrew/bin/cbc", "/usr/local/bin/cbc", shutil.which("cbc")]:
                if p and os.path.exists(p):
                    return pulp.PULP_CBC_CMD(path=p, msg=False)
            try:
                return pulp.PULP_CBC_CMD(msg=False)
            except Exception:
                pass
            try:
                return pulp.apis.HiGHS_CMD(msg=False)
            except Exception:
                return None

        solver = get_solver()

        required_cols = {"Player","Pos","Team","Opp","Sal","Proj"}
        missing_cols = required_cols - set(filtered.columns)
        if missing_cols:
            st.error(f"Optimizer cannot run. Missing columns: {sorted(missing_cols)}")
        elif pulp is None or solver is None:
            st.warning("No MILP solver available. On macOS: `brew install cbc` (recommended) or `pip install highspy`.")

        # -------------------- RUN OPTIMIZER ONLY ON CLICK --------------------
        if generate_clicked and pulp is not None and solver is not None:
            # Build pool from filtered
            pool = filtered.copy()
            pool = pool.dropna(subset=["Player","Pos","Sal","Proj"])
            pool = pool[(pool["Sal"] > 0) & (pool["Pos"].isin(["QB","RB","WR","TE","DST"]))].reset_index(drop=True)

            # Include-only selection
            include_players = sorted(set(sum([st.session_state.get(f"include_{p}", []) for p in ["QB","RB","WR","TE","DST"]], [])))
            if include_players:
                pool = pool[pool["Player"].isin(include_players)].reset_index(drop=True)
            else:
                fallback = base_ui_pool[base_ui_pool["Proj"]>=5]["Player"].tolist()
                pool = pool[pool["Player"].isin(fallback)].reset_index(drop=True)

            # Exclusions
            if exclude_players:
                pool = pool[~pool["Player"].isin(exclude_players)].reset_index(drop=True)

            # Locks
            name_to_idx = {pool.loc[i, "Player"]: i for i in pool.index}
            locked_idx = [name_to_idx[p] for p in lock_players if p in name_to_idx]
            missing_locked = [p for p in lock_players if p not in name_to_idx]
            if missing_locked:
                st.warning("Locked player(s) not in pool and ignored: " + ", ".join(missing_locked))

            # Feasibility checks
            if len(pool) < 9:
                st.error("Not enough players in the pool to build a lineup (need at least 9).")
            else:
                L_qb  = sum(pool.loc[i,"Pos"]=="QB" for i in locked_idx)
                L_rb  = sum(pool.loc[i,"Pos"]=="RB" for i in locked_idx)
                L_wr  = sum(pool.loc[i,"Pos"]=="WR" for i in locked_idx)
                L_te  = sum(pool.loc[i,"Pos"]=="TE" for i in locked_idx)
                L_dst = sum(pool.loc[i,"Pos"]=="DST" for i in locked_idx)
                L_rwt = L_rb + L_wr + L_te
                feasible_lock, msgs = True, []
                if L_qb>1: feasible_lock=False; msgs+=["You locked more than one QB."]
                if L_dst>1: feasible_lock=False; msgs+=["You locked more than one DST."]
                if L_rwt>7: feasible_lock=False; msgs+=["You locked more than 7 total among RB/WR/TE."]
                if len(locked_idx)>9: feasible_lock=False; msgs+=["You locked more than 9 total players."]
                rem_rwt = 7 - L_rwt
                deficit_rb = max(0, 2 - L_rb); deficit_wr = max(0, 3 - L_wr); deficit_te = max(0, 1 - L_te)
                if rem_rwt < (deficit_rb + deficit_wr + deficit_te):
                    feasible_lock=False; msgs+=["Locked RB/WR/TE mix makes per-position minimums impossible."]

                if not feasible_lock:
                    st.error("Lock selection infeasible: " + " ".join(msgs))
                else:
                    allowed_per_player = math.ceil(max_exposure_pct / 100.0 * n_lineups)
                    if allowed_per_player == 0 and n_lineups > 0 and len(locked_idx) == 0:
                        st.error("Max exposure of 0% with ≥1 lineup is infeasible. Increase exposure or reduce # of lineups.")
                    else:
                        rng = np.random.default_rng()
                        lineups = []
                        banned_lineups = []
                        used_counts = {i: 0 for i in pool.index}

                        idx_all = pool.index.tolist()
                        idx_qb  = pool.index[pool["Pos"]=="QB"].tolist()
                        idx_rb  = pool.index[pool["Pos"]=="RB"].tolist()
                        idx_wr  = pool.index[pool["Pos"]=="WR"].tolist()
                        idx_te  = pool.index[pool["Pos"]=="TE"].tolist()
                        idx_dst = pool.index[pool["Pos"]=="DST"].tolist()
                        idx_rwt = pool.index[pool["Pos"].isin(["RB","WR","TE"])].tolist()

                        feasible_positions = (len(idx_qb)>=1 and len(idx_rb)>=2 and len(idx_wr)>=3 and
                                            len(idx_te)>=1 and len(idx_dst)>=1 and len(idx_rwt)>=7)
                        if not feasible_positions:
                            st.error("Pool does not have enough players per position after includes/excludes/locks.")
                        else:
                            def build_one_lineup(scrambled_proj, banned_sets, banned_player_idx, locked_idx_in):
                                prob = pulp.LpProblem("DK_NFL_Optimizer", pulp.LpMaximize)
                                x = pulp.LpVariable.dicts("x", idx_all, lowBound=0, upBound=1, cat="Binary")

                                prob += pulp.lpSum(scrambled_proj[i] * x[i] for i in idx_all)
                                prob += pulp.lpSum(pool.loc[i, "Sal"] * x[i] for i in idx_all) <= dk_salary_cap
                                prob += pulp.lpSum(x[i] for i in idx_qb)  == 1
                                prob += pulp.lpSum(x[i] for i in idx_rb)  >= 2
                                prob += pulp.lpSum(x[i] for i in idx_wr)  >= 3
                                prob += pulp.lpSum(x[i] for i in idx_te)  >= 1
                                prob += pulp.lpSum(x[i] for i in idx_dst) == 1
                                prob += pulp.lpSum(x[i] for i in idx_rwt) == 7
                                prob += pulp.lpSum(x[i] for i in idx_all) == 9

                                for chosen_set in banned_sets:
                                    prob += pulp.lpSum([x[i] for i in chosen_set]) <= 8
                                for i in banned_player_idx:
                                    prob += x[i] == 0
                                for i in locked_idx_in:
                                    prob += x[i] == 1

                                prob.solve(solver)
                                status = pulp.LpStatus[prob.status]
                                if status != "Optimal":
                                    return status, None
                                chosen = [i for i in idx_all if pulp.value(x[i]) == 1]
                                return status, chosen

                            attempts, max_attempts = 0, n_lineups * 6
                            while len(lineups) < n_lineups and attempts < max_attempts:
                                attempts += 1
                                banned_player_idx = [i for i, c in used_counts.items()
                                                    if (i not in locked_idx) and (allowed_per_player > 0) and (c >= allowed_per_player)]
                                v = variance_pct / 100.0
                                multipliers = rng.uniform(1.0 - v, 1.0 + v, size=len(pool))
                                scrambled = pd.Series(pool["Proj"].values * multipliers, index=pool.index)

                                status, chosen = build_one_lineup(scrambled, banned_lineups, banned_player_idx, locked_idx)
                                if status != "Optimal" or chosen is None:
                                    continue
                                chosen_set = frozenset(chosen)
                                if chosen_set in banned_lineups:
                                    continue

                                for i in chosen:
                                    used_counts[i] += 1
                                lu = pool.loc[chosen, ["Player","Pos","Team","Opp","Sal","Proj"]].copy()
                                lu["Proj Used"] = scrambled.loc[chosen].round(2)
                                lineups.append({
                                    "players_df": lu.copy(),
                                    "total_proj": float(lu["Proj Used"].sum()),
                                    "total_sal": int(lu["Sal"].sum()),
                                    "scrambled_series": scrambled
                                })
                                banned_lineups.append(chosen_set)

                            # Save results (or show errors)
                            if len(lineups) == 0:
                                st.error("No feasible lineups found. Loosen exposure/locks, widen includes, or reduce variance.")
                                st.session_state.pop("dfs_results", None)
                            else:
                                # ---- Totals table
                                totals = pd.DataFrame(
                                    [{"Lineup #": i+1,
                                    "Total Salary": f"${lu['total_sal']:,.0f}",
                                    "Total Proj (scrambled)": round(lu["total_proj"], 2)}
                                    for i, lu in enumerate(lineups)]
                                )

                                # ---- Details table
                                details_rows, upload_rows_display, missing_names = [], [], set()

                                def dk_id(name: str):
                                    return dk_id_dict.get(name)

                                def lineup_to_row_display(players_df, scrambled_series):
                                    qbs = players_df[players_df["Pos"]=="QB"]["Player"].tolist()
                                    rbs = players_df[players_df["Pos"]=="RB"]["Player"].tolist()
                                    wrs = players_df[players_df["Pos"]=="WR"]["Player"].tolist()
                                    tes = players_df[players_df["Pos"]=="TE"]["Player"].tolist()
                                    dst = players_df[players_df["Pos"]=="DST"]["Player"].tolist()
                                    rb_main, wr_main, te_main = rbs[:2], wrs[:3], tes[:1]
                                    extras = []
                                    if len(rbs) > 2: extras += rbs[2:]
                                    if len(wrs) > 3: extras += wrs[3:]
                                    if len(tes) > 1: extras += tes[1:]
                                    if extras:
                                        name_to_idx2 = {pool.loc[idx,"Player"]: idx for idx in pool.index}
                                        flex = max(extras, key=lambda nm: scrambled_series[name_to_idx2[nm]])
                                    else:
                                        flex = wr_main[-1] if wr_main else ""
                                    row_names = {
                                        "QB":  qbs[0] if qbs else "",
                                        "RB":  rb_main[0] if len(rb_main)>0 else "",
                                        "RB2": rb_main[1] if len(rb_main)>1 else "",
                                        "WR":  wr_main[0] if len(wr_main)>0 else "",
                                        "WR2": wr_main[1] if len(wr_main)>1 else "",
                                        "WR3": wr_main[2] if len(wr_main)>2 else "",
                                        "TE":  te_main[0] if te_main else "",
                                        "FLEX": flex,
                                        "DST": dst[0] if dst else "",
                                    }
                                    row_ids = {slot: dk_id(pname) for slot, pname in row_names.items()}
                                    row_display = {}
                                    for slot in ["QB","RB","RB2","WR","WR2","WR3","TE","FLEX","DST"]:
                                        name = row_names.get(slot, "")
                                        pid  = row_ids.get(slot)
                                        if name and pid:
                                            row_display[slot] = f"{name} ({pid})"
                                        elif name:
                                            row_display[slot] = name
                                        else:
                                            row_display[slot] = ""
                                    return row_names, row_ids, row_display

                                for k, lu in enumerate(lineups, start=1):
                                    players_df = lu["players_df"].sort_values(["Pos","Player"]).copy()
                                    players_df["Lineup #"] = k
                                    players_df["DK ID"] = players_df["Player"].map(lambda n: dk_id_dict.get(n))
                                    details_rows.append(players_df)

                                    row_names, row_ids, row_display = lineup_to_row_display(players_df, lu["scrambled_series"])
                                    for slot, pid in row_ids.items():
                                        if row_names[slot] and (pid is None):
                                            missing_names.add(row_names[slot])
                                    upload_rows_display.append({**row_display, "Lineup #": k})

                                out_df = pd.concat(details_rows, ignore_index=True)
                                out_df_display = out_df.copy()
                                out_df_display["Sal"] = out_df_display["Sal"].map(lambda x: f"${x:,.0f}")
                                out_df_display = out_df_display[["Lineup #","Player","DK ID","Pos","Team","Opp","Sal","Proj","Proj Used"]]

                                upload_df = pd.DataFrame(upload_rows_display).sort_values("Lineup #")
                                upload_df = upload_df[["QB","RB","RB2","WR","WR2","WR3","TE","FLEX","DST"]]
                                csv_buf = io.StringIO()
                                upload_df.to_csv(csv_buf, index=False)
                                

                                # Cache results so UI changes don't recompute
                                import datetime as dt 

                                st.session_state["dfs_results"] = {
                                    "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "totals": totals,
                                    "details": out_df_display,
                                    "upload_df": upload_df,
                                    "upload_csv": csv_buf.getvalue(),
                                    "missing_names": sorted(missing_names),
                                    "used_counts": used_counts,
                                    "pool_meta": pool[["Player","Pos"]].copy()
                                }

        # -------------------- DISPLAY LAST GENERATED RESULT (if any) --------------------
        if "dfs_results" in st.session_state:
            res = st.session_state["dfs_results"]
            if res.get("generated_at"):
                st.caption(f"Last generated: {res['generated_at']}")

            st.markdown("#### Lineup Totals")
            totcol1, totcol2, totcol3 = st.columns([1,1,1])
            with totcol2:
                st.dataframe(res["totals"], use_container_width=True, hide_index=True)

            st.markdown("#### Lineup Details")
            st.dataframe(res["details"], use_container_width=True, hide_index=True, height=420)

            if res.get("missing_names"):
                st.warning(
                    "Some players were missing DK IDs in your `dk_id_dict` and appear without an ID in the CSV: "
                    + ", ".join(res["missing_names"])
                )

            st.download_button(
                "Download DK Upload CSV (Name + ID)",
                data=res["upload_csv"],
                file_name="dk_lineups.csv",
                mime="text/csv"
            )

            # Exposure table (optional, computed from cached counts)
            show_exposure = st.checkbox("Show player exposure table")
            if show_exposure:
                used_counts = res["used_counts"]
                pool_meta = res["pool_meta"]
                exp_df = (
                    pd.DataFrame({"idx": list(used_counts.keys()), "Times Used": list(used_counts.values())})
                    .merge(pool_meta.reset_index().rename(columns={"index":"idx"}), on="idx", how="left")
                    .drop(columns=["idx"])
                )
                # Only show players used at least once
                exp_df = exp_df[exp_df["Times Used"] > 0].copy()
                exp_df["Exposure %"] = (exp_df["Times Used"] / max(len(res["details"]["Lineup #"].unique()), 1) * 100).round(1)
                exp_df = exp_df.sort_values(["Exposure %","Times Used","Player"], ascending=[False,False,True])
                st.markdown("#### Player Exposure")
                expcol1, expcol2, expcol3 = st.columns([1,2,1])
                with expcol2:
                    pos_options = ["All"] + sorted(exp_df["Pos"].dropna().unique().tolist())
                    pos_choice = st.selectbox("Filter by position:", pos_options, index=0, key="exp_pos_filter")

                    exp_view = exp_df if pos_choice == "All" else exp_df[exp_df["Pos"] == pos_choice]

                    st.dataframe(
                        exp_view[["Player", "Pos", "Times Used", "Exposure %"]],
                        height=900, use_container_width=True, hide_index=True
                    )




    if tab == "Player Grades":
        # ---------- Header ----------
        st.markdown(
            """<div style="text-align:center; line-height:1.3">
                <div style="font-family:Futura; font-weight:800; font-size:44px;">Follow The Money Player Grades</div>
                <div style="font-family:Futura; font-size:14px; opacity:.8;">
                    Algorithmic player rankings using inputs that are more important to fantasy football success
                </div>
            </div><br>""",
            unsafe_allow_html=True,
        )

        # ---------- Helper to prep each position df ----------
        def prep_pos_df(pos: str) -> pd.DataFrame:
            src = {"QB": qb_grades, "RB": rb_grades, "WR": wr_grades, "TE": te_grades}[pos].copy()

            # Normalize types
            src["Season"] = pd.to_numeric(src["Season"], errors="coerce")
            # Map position-specific name column to a unified "Player"
            name_col = pos
            src = src.rename(columns={name_col: "Player"})
            # Identify week columns if present (1,2,3 ... or "1","2",...)
            week_cols = []
            for c in src.columns:
                if isinstance(c, (int, np.integer)):
                    week_cols.append(int(c))
                elif isinstance(c, str) and c.isdigit():
                    week_cols.append(int(c))
            week_cols = sorted(list(set(week_cols)))
            # Keep only sensible columns
            keep_cols = ["Player", "Team", "Season"] + [str(w) for w in week_cols]
            keep_cols = [c for c in keep_cols if c in src.columns]
            src = src[keep_cols].dropna(subset=["Player", "Team", "Season"])
            return src, week_cols

        # ---------- Controls ----------
        c1, c2, c3, c4 = st.columns([1, 1, 1.4, 1.6])
        with c1:
            select_pos = st.selectbox("Position", ["QB", "RB", "WR", "TE"])
        df_raw, all_weeks = prep_pos_df(select_pos)

        # Reasonable default min grade threshold (your old code used >10 for RB/WR/TE)
        default_min = 0 if select_pos == "QB" else 10
        with c2:
            min_grade = st.slider("Min season grade", 0, 100, int(default_min), step=1)
        with c3:
            team_options = ["All"] + sorted(df_raw["Team"].unique().tolist())
            team_choice = st.selectbox("Team filter", team_options, index=0)
        with c4:
            search_name = st.text_input("Search player", placeholder="Type a name…")

        # Week selection row
        show_weekly = st.checkbox("Show weekly grades", value=False)
        chosen_weeks = []
        if show_weekly and all_weeks:
            # Default: last 4 weeks (or all if <4)
            default_weeks = all_weeks[-min(4, len(all_weeks)):]
            chosen_weeks = st.multiselect(
                "Weeks to show",
                options=all_weeks,
                default=default_weeks,
                help="These columns will be added to the table and colored independently."
            )
            chosen_weeks = [str(w) for w in chosen_weeks if str(w) in df_raw.columns]

        # Top-N selector
        top_n = st.slider("Show top N", 10, 300, min(50, len(df_raw)), step=5)

        # ---------- Filtering ----------
        df = df_raw.copy()
        df = df[df["Season"] >= min_grade]
        if team_choice != "All":
            df = df[df["Team"] == team_choice]
        if search_name:
            df = df[df["Player"].str.contains(search_name, case=False, na=False)]

        # Rank & sort
        df = df.sort_values("Season", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", np.arange(1, len(df) + 1))
        display_cols = ["Rank", "Player", "Team", "Season"] + chosen_weeks
        df_display = df[display_cols].head(top_n)

        # ---------- Styling (2 decimals + gradients) ----------
        def style_table(df_show: pd.DataFrame) -> pd.io.formats.style.Styler:
            numeric_cols = df_show.select_dtypes(include="number").columns.tolist()
            fmt = {col: "{:.2f}" for col in numeric_cols}

            styler = df_show.style.format(fmt)
            # Green (good) -> red (bad). Use 'RdYlGn' reversed for "higher=better"
            try:
                styler = styler.background_gradient(
                    cmap="RdYlGn_r", subset=["Season"]
                )
            except Exception:
                pass
            for wk in chosen_weeks:
                if wk in df_show.columns:
                    try:
                        styler = styler.background_gradient(cmap="RdYlGn_r", subset=[wk])
                    except Exception:
                        pass

            # Make Rank column tighter & bold
            def bold_rank(col):
                return ["font-weight:700; width:1px;"] * len(col)
            if "Rank" in df_show.columns:
                styler = styler.set_properties(subset=["Rank"], **{"font-weight": "700", "text-align": "center"})

            # Align text
            styler = styler.set_properties(subset=["Player", "Team"], **{"white-space": "nowrap"})
            styler = styler.set_table_styles([
                {"selector": "th", "props": [("font-weight", "700"), ("text-transform", "uppercase")]},
                {"selector": "thead tr th", "props": [("text-align", "center")]},
            ])
            return styler

        tcol1, tcol2, tcol3 = st.columns([1, 3, 1])
        with tcol2:
            st.dataframe(
                style_table(df_display),
                use_container_width=True,
                hide_index=True,
                height=700 if not show_weekly else 900
            )

        # ---------- Quick chart of top players ----------
        if len(df_display) >= 3:
            st.markdown("##### Top players (Season grade)")
            chart_df = df_display[["Player", "Season"]].set_index("Player").head(15)
            st.bar_chart(chart_df)

        # ---------- Download ----------
        csv = df_display.to_csv(index=False)
        st.download_button(
            "Download current table (CSV)",
            data=csv,
            file_name=f"player_grades_{select_pos.lower()}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # ---------- Light UI polish ----------
        st.markdown(
            """
            <style>
            /* soften the container a bit */
            .stDataFrame thead tr th { text-align:center; }
            .stDataFrame tbody tr td { vertical-align: middle; }
            .st-emotion-cache-10trblm p { margin: 0; }
            </style>
            """,
            unsafe_allow_html=True,
        )


























    if tab == "Player Grades2":

        st.markdown(f"""<br><center><font size=10 face=Futura><b>Follow The Money Player Grades<br></b><font size=3 face=Futura>Algorithmic player rankings using inputs that are more important to fantasy football success</font></center>""", unsafe_allow_html=True)

        gradebox1, gradebox2 = st.columns([1, 3])
        with gradebox1:
            select_pos = st.selectbox('Select a Position', ['QB', 'RB', 'WR', 'TE'])
            #show_weekly = st.checkbox('Show Weekly Data', value=False)

        if select_pos == 'QB':
            # Assuming qb_grades is your DataFrame
            df = qb_grades.copy()
            df['Season'] = df['Season'].astype(int)

            # Add rank column based on Season grade
            df = df.sort_values('Season', ascending=False)
            df['Rank'] = range(1, len(df) + 1)

            # Select columns based on checkbox
            #if not show_weekly:
            df_display = df[['Rank', 'QB', 'Team', 'Season']].copy()
            #else:
                #df_display = df[['Rank', 'QB', 'Team', 'Season', 1, 2, 3]].copy()           

            # Display the styled DataFrame
            showcol1,showcol2,showcol3 = st.columns([1,2,1])
            with showcol2: 
                st.dataframe(df_display, use_container_width=False, width=800, height=1100, hide_index=True)

            # Add some visual flair
            st.markdown("<style>body {background-color: #white;}</style>", unsafe_allow_html=True)
            st.markdown("<style>.stApp {background-color: #white;}</style>", unsafe_allow_html=True)
        
        if select_pos == 'RB':
            # Assuming qb_grades is your DataFrame
            df = rb_grades.copy()
            df = df[df['Season']>10]

            df['Season'] = df['Season'].astype(int)

            # Add rank column based on Season grade
            df = df.sort_values('Season', ascending=False)
            df['Rank'] = range(1, len(df) + 1)

            # Select columns based on checkbox
            #if not show_weekly:
            df_display = df[['Rank', 'RB', 'Team', 'Season']].copy()
            #else:
                #df_display = df[['Rank', 'QB', 'Team', 'Season', 1, 2, 3]].copy()           

            # Display the styled DataFrame
            showcol1,showcol2,showcol3 = st.columns([1,2,1])
            with showcol2: 
                st.dataframe(df_display, use_container_width=False, width=800, height=1100, hide_index=True)

            # Add some visual flair
            st.markdown("<style>body {background-color: #white;}</style>", unsafe_allow_html=True)
            st.markdown("<style>.stApp {background-color: #white;}</style>", unsafe_allow_html=True)
        if select_pos == 'WR':
            # Assuming qb_grades is your DataFrame
            df = wr_grades.copy()
            df = df[df['Season']>10]

            df['Season'] = df['Season'].astype(int)

            # Add rank column based on Season grade
            df = df.sort_values('Season', ascending=False)
            df['Rank'] = range(1, len(df) + 1)

            # Select columns based on checkbox
            #if not show_weekly:
            df_display = df[['Rank', 'WR', 'Team', 'Season']].copy()
            #else:
                #df_display = df[['Rank', 'QB', 'Team', 'Season', 1, 2, 3]].copy()           

            # Display the styled DataFrame
            showcol1,showcol2,showcol3 = st.columns([1,2,1])
            with showcol2: 
                st.dataframe(df_display, use_container_width=False, width=800, height=1100, hide_index=True)

            # Add some visual flair
            st.markdown("<style>body {background-color: #white;}</style>", unsafe_allow_html=True)
            st.markdown("<style>.stApp {background-color: #white;}</style>", unsafe_allow_html=True)
        if select_pos == 'TE':
            # Assuming qb_grades is your DataFrame
            df = te_grades.copy()
            df = df[df['Season']>10]

            df['Season'] = df['Season'].astype(int)

            # Add rank column based on Season grade
            df = df.sort_values('Season', ascending=False)
            df['Rank'] = range(1, len(df) + 1)

            # Select columns based on checkbox
            #if not show_weekly:
            df_display = df[['Rank', 'TE', 'Team', 'Season']].copy()
            #else:
                #df_display = df[['Rank', 'QB', 'Team', 'Season', 1, 2, 3]].copy()           

            # Display the styled DataFrame
            showcol1,showcol2,showcol3 = st.columns([1,2,1])
            with showcol2: 
                st.dataframe(df_display, use_container_width=False, width=800, height=8500, hide_index=True)

            # Add some visual flair
            st.markdown("<style>body {background-color: #white;}</style>", unsafe_allow_html=True)
            st.markdown("<style>.stApp {background-color: #white;}</style>", unsafe_allow_html=True)

    
    if tab == "Book Based Proj":
        st.markdown(f"""<br><center><font size=10 face=Futura><b>Book Based Projections<br></b><font size=3 face=Futura>These are projections derived from the betting lines taken out of the major sports books</font></center>
                     """, unsafe_allow_html=True)
        
        game_select_list = ['All'] + list(bookproj['Game'].unique())
        player_select_list = ['All'] + list(bookproj['Player'])

        bookcol1, bookcol2 = st.columns([1,1])
        with bookcol1:
            game_selection_box = st.selectbox('Select a Game', game_select_list)
        with bookcol2:
            player_selection_box = st.selectbox('Select a Player', player_select_list)
        
        if game_selection_box == 'All':
             show_df = bookproj.copy()
        else:
            filtered_df = bookproj[bookproj['Game']==game_selection_box]
            show_df = filtered_df.copy()

        if player_selection_box == 'All':
            pass
        else:
            filtered_df = bookproj[bookproj['Player']==player_selection_box]
            show_df = filtered_df.copy()

        show_df = show_df[['Player','Team','Opp','Game','Pass Att','Pass Yards','Int','Pass TD','Rush Att','Rush Yds','Rec','Rec Yds','Rush Rec TD']]
        st.dataframe(show_df,hide_index=True, width=1250)
    
    if tab == "Expected Fantasy Points":
        st.markdown("<h1><center>Expected Fantasy Points Model</h1></center>", unsafe_allow_html=True)
        st.markdown("<i><center><h3>work in progress...</i></h3></center>", unsafe_allow_html=True)
        st.markdown("<center><a href='https://docs.google.com/spreadsheets/d/106Wj9ncKwDnfDuUQUc5CHHX3sg_hE0Fkm4L75H_jC0E/edit?usp=sharing'>View Model Results Here</a></center><br>",unsafe_allow_html=True)
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

        g_checkcol1, g_checkcol2, g_checkcol3 = st.columns([1,1,4])

        with g_checkcol1:
            show_all_game_info = st.checkbox('Show Full Slate Game Info', value=False)
        with g_checkcol2:
            show_shootout_info = st.checkbox('Show Shootout Game Info', value=False)
        if show_all_game_info:
            show_schedule = implied_totals[['Team','Opp','OU','Spread','Implied']].sort_values(by='OU',ascending=False)
            show_schedule['MainSlate'] = np.where(show_schedule['Team'].isin(main_slate_team_list),"Y","N")
            scol1, scol2, scol3 = st.columns([1,1,1])
            with scol2:
                st.dataframe(show_schedule, width=800,height=900, hide_index=True)
        if show_shootout_info:
            show_shooty_matchups = shootout_matchups[['Team','Opp','Game SS']]
            show_shooty_matchups['Game SS'] = show_shooty_matchups['Game SS'].astype(int)
            show_shooty_matchups.columns=['Team','Opp','Game Shootout Score']
            sscol1, sscol2, sscol3 = st.columns([1,1,1])
            with sscol2:
                st.dataframe(show_shooty_matchups, width=460, height=1200, hide_index=True)

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
        from datetime import datetime
        check_today = datetime.today().date()
        this_week = this_week[this_week['Date']>=check_today]
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
        game_ss_value = game_ss_dict.get(road_team_short)
        game_ss_value = int(game_ss_value)

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
                        <font size=6 face=Futura>Over/Under: {game_ou}</font><br>
                        <font size=6 face=Futura>Shootout Score: {game_ss_value}</font><br>
                        <font size=3 face=Arial><i>100 = league average shootout score</i></font></center>
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
        weekproj['JA Rk'] = weekproj['Player'].map(all_grade_rank_dict)
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
                road_def_grade = team_grades[team_grades['Team']==road_team_short]['Defense Grade'].iloc[0].astype(int)
                road_off_grade = team_grades[team_grades['Team']==road_team_short]['Offense Grade'].iloc[0].astype(int)
                home_def_grade = team_grades[team_grades['Team']==home_team_short]['Defense Grade'].iloc[0].astype(int)
                home_off_grade = team_grades[team_grades['Team']==home_team_short]['Offense Grade'].iloc[0].astype(int)

                st.markdown(f"<center><font size=13><b>{road_team}</b></font><br><font size=4><i>Implied for <font size=6 color=red><b>{road_implied}</b></font> points, ranked #<font size = 6 color=red><b>{int(road_implied_rank)}</b></font> of {len(implied_totals)}</i><br><i>D Grade: <b><font size=6 color=red>{road_def_grade}</font></b>   |   O Grade: <b><font size = 6 color=red>{road_off_grade}</font></b></i><hr>", unsafe_allow_html=True)

                st.markdown("<h4>Quarterback</h4>",unsafe_allow_html=True)
                road_qb_proj = road_projections[road_projections['Pos']=='QB'][['Player','JA Rk','Sal','Pass Comp','Pass Att','Pass Yards','Pass TD', 'Int','Rush Att','Rush Yds','Rush TD']].sort_values(by='Pass Att',ascending=False)
                st.dataframe(road_qb_proj, hide_index=True, width=750)
                st.markdown("<h4>Running Backs</h4>",unsafe_allow_html=True)
                road_rb_proj = road_projections[road_projections['Pos']=='RB'][['Player','JA Rk','Sal','Rush Att','Rush Yds','Rush TD','Tgt','Rec','Rec Yds','Rec TD']].sort_values(by='Rush Att',ascending=False)
                st.dataframe(road_rb_proj, hide_index=True, width=650,height=150)
                st.markdown("<h4>Pass Catchers</h4>",unsafe_allow_html=True)
                road_rec_proj = road_projections[road_projections['Pos'].isin(['WR','TE'])][['Player','JA Rk','Sal','Tgt','Rec','Rec Yds','Rec TD']].sort_values(by='Rec Yds',ascending=False)
                if len(road_rec_proj) > 7:
                    st.dataframe(road_rec_proj, hide_index=True, width=650,height=325)
                else:
                    st.dataframe(road_rec_proj, hide_index=True, width=650)

            with projcol2:
                #st.markdown(f"<center><font size=13><b>{home_team}</b></font><br><font size=4><i>Implied for {home_implied} points, ranked #{int(home_implied_rank)} of {len(implied_totals)}</i></center><hr>", unsafe_allow_html=True)
                st.markdown(f"<center><font size=13><b>{home_team}</b></font><br><font size=4><i>Implied for <font size=6 color=red><b>{home_implied}</b></font> points, ranked #<font size = 6 color=red><b>{int(home_implied_rank)}</b></font> of {len(implied_totals)}</i><br><i>D Grade: <b><font size=6 color=red>{home_def_grade}</font></b>   |   O Grade: <b><font size = 6 color=red>{home_off_grade}</font></b></i><hr>", unsafe_allow_html=True)

                st.markdown("<h4>Quarterback</h4>",unsafe_allow_html=True)
                home_qb_proj = home_projections[home_projections['Pos']=='QB'][['Player','JA Rk','Sal','Pass Comp','Pass Att','Pass Yards','Pass TD', 'Int','Rush Att','Rush Yds','Rush TD']].sort_values(by='Pass Att',ascending=False)
                st.dataframe(home_qb_proj, hide_index=True, width=750)
                st.markdown("<h4>Running Backs</h4>",unsafe_allow_html=True)
                home_rb_proj = home_projections[home_projections['Pos']=='RB'][['Player','JA Rk','Sal','Rush Att','Rush Yds','Rush TD','Tgt','Rec','Rec Yds','Rec TD']].sort_values(by='Rush Att',ascending=False)
                st.dataframe(home_rb_proj, hide_index=True, width=650,height=150)
                st.markdown("<h4>Pass Catchers</h4>",unsafe_allow_html=True)
                home_rec_proj = home_projections[home_projections['Pos'].isin(['WR','TE'])][['Player','JA Rk','Sal','Tgt','Rec','Rec Yds','Rec TD']].sort_values(by='Rec Yds',ascending=False)
                if len(home_rec_proj) > 7:
                    st.dataframe(home_rec_proj, hide_index=True, width=650,height=325)
                else:
                    st.dataframe(home_rec_proj, hide_index=True, width=650)
    if tab == "Line Movement":
        st.write(this_week_schedule.sort_values(by=['Home','Timestamp']))
    
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

            #dfs_sals_check = st.checkbox('Show DFS Info?', value=True)
            dfs_sals_dict = dict(zip(dkdata.Player,dkdata.Sal))
            dfs_sals_check = True

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
            
            ## current way
            #st.dataframe(show_filtered_data, hide_index=True, height=750)

            ## new way
            # Apply a gradient to the Age column and bold text for Status
            columns_to_round = ['FPts', 'Val', 'Pos Rank', 'Pass Comp', 'Pass Att', 'Pass Yards', 'Pass TD', 'Int', 'Rush Att']

            show_filtered_data_reset = show_filtered_data.reset_index(drop=True)

            styled_df = show_filtered_data.style.background_gradient(subset=['FPts'], cmap='Blues')\
            .set_properties(**{'font-weight': 'bold'}, subset=['FPts'])\
            .format({
                'FPts': '{:.1f}', 'Own': '{:.0f}',
                'Val': '{:.1f}','Sal': '{:.0f}',
                'Pos Rank': '{:.1f}','Rush Yds': '{:.1f}',
                'Pass Comp': '{:.1f}','Rush TD': '{:.1f}',
                'Pass Att': '{:.1f}','Tgt': '{:.1f}',
                'Pass Yards': '{:.1f}','Rec': '{:.1f}',
                'Pass TD': '{:.1f}','Rec Yds': '{:.1f}',
                'Int': '{:.1f}','Rec TD': '{:.1f}',
                'Rush Att': '{:.1f}'
            })
            
            #st.table(styled_df)
            #st.dataframe(show_filtered_data_reset, hide_index=True, height=1000,use_container_width=True)
            
            df = show_filtered_data_reset.copy()

            # -------- helpers --------
            def safe_minmax(s: pd.Series, pad=0.0):
                """Return (vmin, vmax) with a tiny spread if the column is constant."""
                vmin, vmax = float(s.min()), float(s.max())
                if vmin == vmax:
                    eps = 1e-9 if vmin == 0 else abs(vmin) * 1e-6
                    vmin, vmax = vmin - eps, vmax + eps
                return vmin - pad, vmax + pad

            # Independent ranges (each column scaled only to itself)
            vmin_fpts, vmax_fpts = safe_minmax(df["FPts"])
            vmin_val,  vmax_val  = safe_minmax(df["Val"])

            # -------- table styles (readability) --------
            table_styles = [
                {"selector": "th.col_heading",
                "props": [("font-weight", "700"), ("font-size", "14px"),
                        ("background-color", "#f6f8fa"), ("color", "#111827"),
                        ("border-bottom", "1px solid #e5e7eb")]},
                {"selector": "td",
                "props": [("font-size", "14px"), ("color", "#0f172a"),
                        ("font-family", "Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif"),
                        ("white-space", "nowrap")]},
                {"selector": "tbody tr:hover",
                "props": [("background-color", "#fafafa")]},
            ]

            # -------- formatting: 2 decimals globally, special cases override --------
            fmt = {col: "{:.2f}" for col in df.select_dtypes(include="number").columns}
            if "Sal" in df:       fmt["Sal"] = "${:,.0f}"   # currency, no decimals
            if "Pos Rank" in df:  fmt["Pos Rank"] = "{:.0f}"  # integer, no decimals

            styler = (
                df.style
                .format(fmt)
                .set_table_styles(table_styles)
                .hide(axis="index")
                # color gradients (independent & softened)
                .background_gradient(
                    cmap="RdYlGn", subset=["FPts"], vmin=vmin_fpts, vmax=vmax_fpts,
                    low=0.15, high=0.85
                )
            )

            st.dataframe(
                styler,
                use_container_width=True,
                height=1000,
                hide_index=True
            )




            ##############################



            
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