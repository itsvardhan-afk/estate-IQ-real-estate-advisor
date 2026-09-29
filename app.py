from pathlib import Path

import itertools



import joblib

import numpy as np

import pandas as pd

import streamlit as st





BASE = Path(__file__).resolve().parent

CLASSIFIER_PATH = BASE / "classification_model.pkl"

REGRESSOR_PATH = BASE / "regression_model.pkl"

DATA_PATH = BASE / "india_housing_prices.csv"

REFERENCE_YEAR = 2025
FEATURE_CONTRACT = [
    "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
    "Price_in_Lakhs", "Price_per_SqFt", "Year_Built", "Furnished_Status",
    "Floor_No", "Total_Floors", "Age_of_Property", "Nearby_Schools",
    "Nearby_Hospitals", "Public_Transport_Accessibility", "Parking_Space",
    "Security", "Amenities", "Facing", "Owner_Type", "Availability_Status",
    "Large_Property", "Available_Property",
]



st.set_page_config(page_title="EstateIQ — Property Investment Research Terminal", page_icon="E", layout="wide", initial_sidebar_state="collapsed")



st.markdown("""

<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');

:root{--bg:#0b1220;--panel:#0d1726;--raised:#111e30;--line:#1b2a3d;--text:#e7edf5;--muted:#8190a4;--dim:#506176;--mint:#52e2bd;--mint-dark:#123b39;--mono:'DM Mono',monospace;--sans:'DM Sans',sans-serif}

html,body,[class*="css"]{font-family:var(--sans);color:var(--text)}

.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="stMainBlockContainer"]{background:var(--bg)!important;color:var(--text)!important}

[data-testid="stHeader"]{background:var(--bg)!important;height:0;visibility:hidden}

[data-testid="stSidebar"],footer{display:none!important}

[data-testid="stMainBlockContainer"]{max-width:1250px;padding:0 2.2rem 4rem!important}

h1,h2,h3,h4,p{color:var(--text)}

.topbar{height:75px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;margin:0 -2.2rem 3.2rem;padding:0 2.2rem;}

.brand{display:flex;align-items:center;gap:11px;color:#eaf0f7;font-weight:700;font-size:15px;letter-spacing:-.3px}

.brand-mark{display:flex;align-items:center;justify-content:center;width:28px;height:28px;border:1px solid #24464a;border-radius:6px;background:#10232b;color:var(--mint);font-weight:700;font-size:14px}

.top-meta{font-family:var(--mono);color:var(--dim);font-size:10px;letter-spacing:1px}

.eyebrow{font-family:var(--mono);font-size:10px;line-height:1.5;letter-spacing:1.5px;text-transform:uppercase;color:var(--mint);margin:0 0 12px}

.hero-title{font-size:clamp(38px,5vw,58px);font-weight:600;letter-spacing:-2.8px;line-height:1.05;max-width:700px;margin:0 0 16px}

.hero-copy{font-size:15px;line-height:1.7;color:#95a2b4;max-width:660px;margin:0 0 32px}

.tagline{font-family:var(--mono);font-size:10px;letter-spacing:1.5px;color:var(--mint);text-transform:uppercase}

.page-title{font-size:34px;line-height:1.15;letter-spacing:-1.4px;font-weight:600;margin:0 0 10px}

.subtle{color:var(--muted);font-size:13px;line-height:1.65}

.section-label{font-family:var(--mono);font-size:10px;letter-spacing:1.35px;color:var(--mint);text-transform:uppercase}

.section-title{font-size:23px;line-height:1.25;letter-spacing:-.7px;font-weight:600;margin:10px 0 8px}

.card{height:100%;min-height:188px;padding:24px;background:var(--panel);border:1px solid var(--line);border-radius:10px}

.mini-card{height:100%;padding:17px 18px;background:var(--panel);border:1px solid var(--line);border-radius:8px}

.mini-label{font-family:var(--mono);font-size:9px;letter-spacing:1.05px;line-height:1.45;color:#8391a4;text-transform:uppercase;margin-bottom:8px}

.mini-value{font-family:var(--mono);font-size:15px;color:#d9e2ec;line-height:1.4}

.feature-card{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:16px 18px;margin:0 0 9px}

.feature-number{font-family:var(--mono);color:var(--mint);font-size:10px;margin-right:13px}

.feature-copy{font-size:13px;color:#bec9d6}

.notice{font-size:11px;line-height:1.65;color:#66778e;margin-top:20px}

.model-card{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:23px 24px;margin-bottom:15px}

.model-head{font-family:var(--mono);font-size:10px;letter-spacing:1.4px;color:var(--mint);text-transform:uppercase;margin-bottom:8px}

.feature-chip{font-family:var(--mono);font-size:10px;color:#b6c3d2;background:#0b1421;border:1px solid #1d2b3e;border-radius:5px;padding:8px 10px;margin:4px 0;display:block}

[data-testid="stRadio"]>div{gap:7px;justify-content:flex-end}

[data-testid="stRadio"] label{border:1px solid transparent;border-radius:6px;padding:8px 13px;background:transparent;color:#8090a4!important;font-size:12px}

[data-testid="stRadio"] label p{color:inherit!important}

[data-testid="stRadio"] label:has(input:checked){background:#142333;border-color:#20364a;color:#dbe5ef!important}

[data-testid="stRadio"] label:hover{color:#e8eff7!important}

[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"]{font-size:11px!important;color:#9ba9ba!important}

[data-testid="stSelectbox"] [data-baseweb="select"]>div,[data-testid="stNumberInput"] input{background:#101c2b!important;color:#e2eaf3!important;border:1px solid #26374b!important;border-radius:6px!important;min-height:40px}

[data-testid="stSelectbox"] [role="combobox"],[data-testid="stSelectbox"] svg{color:#e2eaf3!important;fill:#b8c6d5!important}

[data-testid="stNumberInput"] button{background:#172538!important;color:#d4deea!important;border-color:#26374b!important}

[data-baseweb="popover"],[role="listbox"],[role="option"]{background:#101c2b!important;color:#e2eaf3!important}

[role="option"]:hover,[aria-selected="true"]{background:#19352f!important}

[data-testid="stSlider"] [role="slider"]{background:var(--mint)!important;border-color:var(--mint)!important}

[data-testid="stButton"] button,[data-testid="stFormSubmitButton"] button{background:var(--mint)!important;color:#08211d!important;border:0!important;border-radius:6px!important;font-weight:700!important;padding:.7rem 1rem!important}

[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px}

[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] p{color:#8998aa!important;font-size:10px!important;text-transform:uppercase;letter-spacing:.9px}

[data-testid="stMetricValue"],[data-testid="stMetricValue"] div{font-family:var(--mono)!important;color:#e5edf5!important;font-size:23px!important}

[data-testid="stAlert"]{background:#111e30!important;border:1px solid #29394e!important;color:#dce5ef!important}

[data-testid="stAlert"] p{color:#cbd6e3!important}

hr{border-color:var(--line)!important} a{color:var(--mint)!important}

@media(max-width:800px){[data-testid="stMainBlockContainer"]{padding:0 1rem 3rem!important}.topbar{margin:0 -1rem 2rem;padding:0 1rem}.hero-title{font-size:40px}}

</style>

""", unsafe_allow_html=True)





@st.cache_resource

def load_models():

    if not CLASSIFIER_PATH.exists() or not REGRESSOR_PATH.exists():

        return None, None

    return joblib.load(CLASSIFIER_PATH), joblib.load(REGRESSOR_PATH)





@st.cache_data

def load_data():

    if DATA_PATH.exists():

        return pd.read_csv(DATA_PATH, low_memory=False)

    return pd.DataFrame()





classifier, regressor = load_models()

data = load_data()

classifier_features = list(getattr(classifier, "feature_names_in_", [])) if classifier else []
regressor_features = list(getattr(regressor, "feature_names_in_", [])) if regressor else []
features = classifier_features or regressor_features
contract_error = None
if classifier and regressor:
    if classifier_features and regressor_features and classifier_features != regressor_features:
        contract_error = "The saved models use different ordered input features."
    elif features and features != FEATURE_CONTRACT:
        contract_error = "The saved model feature order does not match the app's 24-feature contract."
    elif not features:
        # Models without feature_names_in_ can only use the documented feature order.
        features = FEATURE_CONTRACT.copy()
if contract_error:
    classifier, regressor = None, None
    st.error(contract_error + " Predictions are disabled until matching model files are available.")


def options(column, fallback):

    if column in data.columns:

        vals = data[column].dropna().astype(str).unique().tolist()

        if vals:

            return sorted(vals)

    return fallback





def scoped_options(column, filters, fallback):

    view = data

    for key, value in filters.items():

        if key in view.columns:

            view = view[view[key].astype(str) == str(value)]

    values = view[column].dropna().astype(str).unique().tolist() if column in view.columns else []

    return sorted(values) if values else options(column, fallback)





def header(active):

    brand, nav, meta = st.columns([1.05, 1.6, .7])

    with brand:

        st.markdown('<div class="brand" style="height:74px"><span class="brand-mark">E</span>EstateIQ</div>', unsafe_allow_html=True)

    with nav:

        choices = ["Overview", "Analysis", "Model Notes"]

        selected=st.radio("Navigation", choices, index=choices.index(active), horizontal=True, label_visibility="collapsed", key="navigation")

    with meta:

        st.markdown('<div class="top-meta" style="text-align:right">IN</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:#1b2a3d;margin:0 -2.2rem 3.2rem"></div>', unsafe_allow_html=True)

    return selected





def overview():

    st.markdown('<div class="eyebrow">PROPERTY INVESTMENT ADVISOR · INDIA</div><div class="hero-title">A clearer way to assess your next property</div><p class="hero-copy">One property profile, two trained models: an investment signal with its probability, and a five-year value estimate.</p>', unsafe_allow_html=True)

    st.button("Open analysis workspace", type="primary", key="go_analysis", on_click=lambda: st.session_state.update(navigation="Analysis"))

    a, b, c, d = st.columns(4, gap="small")

    for col, label, value, note in [

        (a, "Models connected", "2", "Classifier + regression"),

        (b, "Model features", str(len(features) or 24), "Ordered input contract"),

        (c, "Reference year", str(REFERENCE_YEAR), "Age & price basis"),

        (d, "Value horizon", "5 yrs", "Regression estimate"),

    ]:

        col.markdown(f'<div class="mini-card"><div class="mini-label">{label}</div><div class="mini-value">{value}</div><div class="subtle" style="font-size:11px;margin-top:7px">{note}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:48px'></div><div class='section-label'>HOW IT WORKS</div><div class='section-title'>One profile, two model views</div>", unsafe_allow_html=True)

    left, right = st.columns([1.15, 1], gap="medium")

    with left:

        st.markdown('<div class="card"><div class="section-label">WORKFLOW</div><div style="height:18px"></div><span class="feature-number">01</span><b>Describe the property</b><div class="subtle" style="margin:6px 0 18px 32px">Set location, pricing, building details and amenities.</div><span class="feature-number">02</span><b>Review the assessment</b><div class="subtle" style="margin:6px 0 18px 32px">See the classifier signal and its positive-class probability.</div><span class="feature-number">03</span><b>Explore the value outlook</b><div class="subtle" style="margin:6px 0 0 32px">Compare the asking price against the five-year regression estimate.</div></div>', unsafe_allow_html=True)

    with right:

        st.markdown('<div class="card"><div class="section-label">PROJECT SCOPE</div><div class="section-title">Built for property decisions</div><div class="subtle" style="margin:0 0 16px">A focused view of the signals in the original project.</div><div class="feature-copy">Location and neighbourhood context<br><br>Price and property profile<br><br>Building features and nearby services<br><br>Five-year estimated value<br><br>Model-driven investment signal</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="notice">Educational model outputs are only as reliable as the project data and training process.</div>', unsafe_allow_html=True)





def analysis():

    st.markdown('<div class="eyebrow">ANALYSIS WORKSPACE</div><div class="page-title">Property investment signal</div><div class="subtle" style="margin-bottom:26px">Reference year&nbsp; 2025 &nbsp;·&nbsp; Values in ₹ lakh</div>', unsafe_allow_html=True)

    if not classifier or not regressor:

        st.info("Models not connected. Place classification_model.pkl and regression_model.pkl beside app.py to enable predictions.")

    st.markdown('<div class="model-card"><div class="section-title" style="margin-top:0">Property inputs</div><div class="subtle">4 groups</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">LOCATION</div>', unsafe_allow_html=True)

    x1, x2, x3 = st.columns(3)

    state = x1.selectbox("State", options("State", ["Maharashtra","Karnataka","Delhi","Tamil Nadu","Gujarat","Rajasthan","Uttar Pradesh"]))

    city = x2.selectbox("City", scoped_options("City", {"State":state}, ["Mumbai","Pune","Bangalore","New Delhi","Chennai","Hyderabad"]))

    locality = x3.selectbox("Locality", scoped_options("Locality", {"State":state,"City":city}, [f"Locality_{i}" for i in range(1,501)]))

    st.markdown('<div class="section-label" style="margin-top:17px">PRICE & PROPERTY</div>', unsafe_allow_html=True)

    x1,x2,x3,x4 = st.columns(4)

    ptype=x1.selectbox("Property type", options("Property_Type",["Apartment","Independent House","Villa"]))

    bhk_label=x2.selectbox("Bedrooms",["1 BHK","2 BHK","3 BHK","4 BHK","5 BHK"],index=2)

    size=x3.number_input("Built-up area (sq ft)",min_value=250,max_value=20000,value=1500,step=50)

    price=x4.number_input("Asking price (₹ lakh)",min_value=1.0,max_value=10000.0,value=100.0,step=5.0)

    st.markdown('<div class="section-label" style="margin-top:17px">BUILDING</div>', unsafe_allow_html=True)

    x1,x2,x3,x4 = st.columns(4)

    year=x1.number_input("Year built",min_value=1950,max_value=REFERENCE_YEAR,value=2015,step=1)

    floors=x2.number_input("Total floors",min_value=1,max_value=100,value=10,step=1)

    floor=x3.number_input("Property floor",min_value=0,max_value=int(floors),value=min(3,int(floors)),step=1)

    furnished=x4.selectbox("Furnishing",options("Furnished_Status",["Unfurnished","Semi-furnished","Furnished"]))

    st.markdown('<div class="section-label" style="margin-top:17px">AMENITIES & ACCESS</div>', unsafe_allow_html=True)

    x1,x2,x3,x4 = st.columns(4)

    facing=x1.selectbox("Facing",options("Facing",["East","North","South","West"]))

    availability=x2.selectbox("Availability",options("Availability_Status",["Ready_to_Move","Under_Construction"]))

    transport=x3.selectbox("Public transport",options("Public_Transport_Accessibility",["Low","Medium","High"]),index=1)

    parking=x4.selectbox("Parking",options("Parking_Space",["No","Yes"]),index=1)

    x1,x2,x3,x4 = st.columns(4)

    security=x1.selectbox("Security",options("Security",["No","Yes"]),index=1)

    owner=x2.selectbox("Listed by",options("Owner_Type",["Owner","Broker","Builder"]))

    amenity_fallback=[", ".join(v) for n in range(1,6) for v in itertools.permutations(["Clubhouse","Garden","Gym","Playground","Pool"],n)]

    amenities=x3.selectbox("Amenities",options("Amenities",amenity_fallback),index=0)

    x4.markdown("<div style='height:8px'></div>",unsafe_allow_html=True)

    schools=x1.slider("Nearby schools",min_value=0,max_value=10,value=4)

    hospitals=x2.slider("Nearby hospitals",min_value=0,max_value=10,value=3)

    submitted=st.button("Run investment model",type="primary",use_container_width=True)



    if submitted:

        if not classifier or not regressor:

            st.session_state.assessment = None

            st.warning("No assessment yet — add classification_model.pkl and regression_model.pkl beside app.py to run the trained models.")

        else:

            bhk=int(bhk_label.split()[0])

            row={"State":state,"City":city,"Locality":locality,"Property_Type":ptype,"BHK":bhk,"Size_in_SqFt":int(size),"Price_in_Lakhs":float(price),"Price_per_SqFt":float(price)/float(size),"Year_Built":int(year),"Furnished_Status":furnished,"Floor_No":int(floor),"Total_Floors":int(floors),"Age_of_Property":REFERENCE_YEAR-int(year),"Nearby_Schools":int(schools),"Nearby_Hospitals":int(hospitals),"Public_Transport_Accessibility":transport,"Parking_Space":parking,"Security":security,"Amenities":amenities,"Facing":facing,"Owner_Type":owner,"Availability_Status":availability,"Large_Property":int(bhk>=3),"Available_Property":int(str(availability).strip().lower().replace(" ", "_") in {"available", "ready_to_move"})}

            try:

                cols = features or FEATURE_CONTRACT
                if len(cols) != 24 or set(cols) != set(FEATURE_CONTRACT):
                    raise ValueError("The loaded model must use exactly the documented 24 features.")
                missing = [name for name in cols if name not in row]
                if missing:
                    raise ValueError("Input construction is missing model features: " + ", ".join(missing))
                model_input = pd.DataFrame([[row[name] for name in cols]], columns=cols)
                if model_input.shape != (1, 24) or list(model_input.columns) != list(cols):
                    raise ValueError("Input does not match the model's ordered 24-feature contract.")

                classes=list(classifier.classes_)

                positive_idx=classes.index(1) if 1 in classes else classes.index(True)

                probability=float(classifier.predict_proba(model_input)[0][positive_idx])

                future=float(np.asarray(regressor.predict(model_input)).reshape(-1)[0])

                st.session_state.assessment={"row":row,"probability":probability,"future":future}

            except Exception as exc:

                st.error(f"The trained models could not process these inputs: {exc}")

    result=st.session_state.get("assessment")

    if result:

        row=result["row"]; current=row["Price_in_Lakhs"]; future=max(0,result["future"]); probability=result["probability"]

        st.markdown("<div style='height:30px'></div><div class='section-label'>MODEL OUTPUT</div><div class='section-title'>Assessment</div>",unsafe_allow_html=True)

        c1,c2,c3=st.columns(3)

        c1.metric("Investment signal","Accumulate" if probability>=.5 else "Review")

        c2.metric("Positive-class probability",f"{probability:.1%}")

        c3.metric("Estimated value in 5 years",f"₹{future:,.1f} lakh")

        c1,c2=st.columns([1.2,1])

        with c1:

            st.markdown('<div class="section-title">Five-year value outlook</div><div class="subtle">The chart interpolates between asking price and the model estimate.</div>',unsafe_allow_html=True)

            years=["Today","Year 1","Year 2","Year 3","Year 4","Year 5"]

            vals=[current+(future-current)*i/5 for i in range(6)]

            st.line_chart(pd.DataFrame({"Estimated value (₹ lakh)":vals},index=years),color="#52e2bd",height=260)

        with c2:

            st.markdown('<div class="section-title">Property profile</div>',unsafe_allow_html=True)

            st.markdown(f'<div class="model-card"><div class="subtle">{row["City"]} · {row["State"]}</div><div style="font-size:20px;margin:8px 0">{row["Property_Type"]} · {row["BHK"]} BHK</div><div class="subtle">{row["Size_in_SqFt"]:,} sq ft &nbsp;·&nbsp; {row["Age_of_Property"]} years old<br>₹{current:,.1f} lakh &nbsp;·&nbsp; {row["Amenities"]}</div></div>',unsafe_allow_html=True)

        st.markdown('<div class="notice">Educational model output — not investment advice or a guaranteed return.</div>',unsafe_allow_html=True)

    elif not submitted:

        st.markdown('<div class="model-card" style="margin-top:22px"><div class="section-title">No assessment yet</div><div class="subtle">Complete the property profile and run the model to see the signal, probability and five-year outlook.</div></div>',unsafe_allow_html=True)





def model_notes():

    st.markdown('<div class="eyebrow">MODEL NOTES</div><div class="page-title">How the outputs are produced</div><div class="subtle" style="margin-bottom:27px">A transparent view of the project&#39;s model setup and input contract.</div>',unsafe_allow_html=True)

    left,right=st.columns(2,gap="medium")

    with left:

        st.markdown('<div class="model-card"><div class="model-head">CLASSIFICATION</div><div class="section-title">Investment signal</div><div class="subtle">A logistic regression pipeline with standardised numeric inputs and one-hot encoded categories returns the probability of the positive class. Scores of 50% or higher are shown as Accumulate; lower scores are shown as Review.<br><br>Unseen categories are ignored, exactly as in the trained encoder.</div></div>',unsafe_allow_html=True)

    with right:

        st.markdown('<div class="model-card"><div class="model-head">REGRESSION</div><div class="section-title">Five-year value estimate</div><div class="subtle">A ridge regression pipeline on the same feature set estimates the property value five years out. The chart interpolates between the asking price and that estimate as a visual guide, not as a year-by-year forecast.<br><br>Educational model output — not investment advice or a guaranteed return.</div></div>',unsafe_allow_html=True)

    feature_list=["State","City","Locality","Property_Type","BHK","Size_in_SqFt","Price_in_Lakhs","Price_per_SqFt","Year_Built","Furnished_Status","Floor_No","Total_Floors","Age_of_Property","Nearby_Schools","Nearby_Hospitals","Public_Transport_Accessibility","Parking_Space","Security","Amenities","Facing","Owner_Type","Availability_Status","Large_Property","Available_Property"]

    st.markdown('<div style="height:14px"></div><div class="section-title">Ordered model features</div><div class="subtle">24 inputs</div><div style="height:15px"></div>',unsafe_allow_html=True)

    cols=st.columns(3)

    for i,name in enumerate(feature_list):

        cols[i%3].markdown(f'<span class="feature-chip">{i+1:02d}&nbsp;&nbsp; {name}</span>',unsafe_allow_html=True)





if "navigation" not in st.session_state:

    st.session_state.navigation="Overview"

active=header(st.session_state.navigation)

if active=="Overview": overview()

elif active=="Analysis": analysis()

else: model_notes()
