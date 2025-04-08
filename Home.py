import streamlit as st
from tensortractlab import TensorTractLab
#from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
    page_title="TensorTractLab Home",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
#st.markdown( """ <style> [data-testid="collapsedControl"] { display: none } </style> """, unsafe_allow_html=True, ) 

@st.cache_resource  # 👈 Add the caching decorator
def load_model():
   if st.secrets['HF_TOKEN'] is not None:
      hf_token = st.secrets['HF_TOKEN']
   else:
      hf_token = None
   print("Loading model...")
   model = TensorTractLab(hf_token=hf_token)
   model.eval()
   return model


# Animated SVG using SMIL
animated_svg = """
<div style="width: 900; text-align: center">
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!-- Created with Inkscape (http://www.inkscape.org/) -->

<svg
   width="44.130787mm"
   height="45.447491mm"
   viewBox="0 0 44.130787 45.447491"
   version="1.1"
   id="svg172802"
   inkscape:version="1.2.2 (b0a84865, 2022-12-01)"
   sodipodi:docname="TensorTractLabLogo_try0.svg"
   inkscape:export-filename="TensorTractLabLogo_try0.png"
   inkscape:export-xdpi="96"
   inkscape:export-ydpi="96"
   xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"
   xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
   xmlns:xlink="http://www.w3.org/1999/xlink"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:svg="http://www.w3.org/2000/svg">
  <sodipodi:namedview
     id="namedview172804"
     pagecolor="#ffffff"
     bordercolor="#666666"
     borderopacity="1.0"
     inkscape:pageshadow="2"
     inkscape:pageopacity="0.0"
     inkscape:pagecheckerboard="0"
     inkscape:document-units="mm"
     showgrid="false"
     fit-margin-top="0"
     fit-margin-left="0"
     fit-margin-right="0"
     fit-margin-bottom="0"
     inkscape:zoom="0.58083614"
     inkscape:cx="586.22385"
     inkscape:cy="158.39235"
     inkscape:window-width="1344"
     inkscape:window-height="456"
     inkscape:window-x="0"
     inkscape:window-y="38"
     inkscape:window-maximized="0"
     inkscape:current-layer="layer1"
     inkscape:showpageshadow="2"
     inkscape:deskcolor="#d1d1d1" />
  <defs
     id="defs172799">
    <linearGradient
       inkscape:collect="always"
       id="linearGradient1261">
      <stop
         style="stop-color:#183f5d;stop-opacity:1;"
         offset="0"
         id="stop1255" />
      <stop
         style="stop-color:#2ea8a6;stop-opacity:1;"
         offset="0.73143739"
         id="stop1257" />
      <stop
         style="stop-color:#e4ff00;stop-opacity:1;"
         offset="1"
         id="stop1259" />
    </linearGradient>
    <linearGradient
       inkscape:collect="always"
       xlink:href="#linearGradient1261"
       id="linearGradient8816"
       x1="60.767902"
       y1="180.29715"
       x2="50.351746"
       y2="208.67975"
       gradientUnits="userSpaceOnUse"
       gradientTransform="translate(-120.56094,-53.434959)" />
    <linearGradient
       inkscape:collect="always"
       xlink:href="#linearGradient1261"
       id="linearGradient10728"
       x1="46.247498"
       y1="179.62228"
       x2="37.879955"
       y2="152.82854"
       gradientUnits="userSpaceOnUse"
       gradientTransform="matrix(0.95581618,0,0,0.91577889,-117.61506,-37.970184)" />
    <linearGradient
       inkscape:collect="always"
       xlink:href="#linearGradient1261"
       id="linearGradient1253"
       x1="-76.455437"
       y1="151.77548"
       x2="-102.65134"
       y2="114.37438"
       gradientUnits="userSpaceOnUse" />
  </defs>
  <g
     inkscape:label="Ebene 1"
     inkscape:groupmode="layer"
     id="layer1"
     transform="translate(96.491068,-112.10068)">
    <path
       style="fill:url(#linearGradient8816);fill-opacity:1;stroke-width:0.264583"
       d="m -73.233356,153.54174 v -4.01525 l 3.13031,-5.61999 c 4.90971,-8.81462 5.12785,-9.224 4.98911,-9.36274 -0.072,-0.072 -1.88873,0.76513 -4.03718,1.86029 -2.14844,1.09517 -4.00539,1.92995 -4.12654,1.85507 -0.12115,-0.0749 -0.22028,-1.6268 -0.22028,-3.44872 v -3.14175 l 10.44294,-5.3093 c 5.70615,-2.90107 10.50612,-5.50859 10.58334,-5.50859 0.19272,0 0.11749,5.60548 -0.0893,6.58782 -0.14416,0.68461 -0.52246,0.97728 -2.75125,2.12844 -1.42126,0.73408 -2.80311,1.56752 -3.07078,1.85209 -0.26767,0.28456 -2.07169,3.37489 -4.00895,6.86739 -7.86713,14.18296 -10.54744,18.97656 -10.69075,19.11987 -0.0828,0.0828 -0.15062,-1.65625 -0.15062,-3.86463 z"
       id="path1164-3"
       sodipodi:nodetypes="ccssssscsscsssscc" />
    <path
       style="fill:url(#linearGradient1253);fill-opacity:1;stroke:none;stroke-width:0.270583;stroke-opacity:1"
       d="m -74.660661,153.6807 v -4.01525 l -3.273878,-5.61999 c -5.134895,-8.81462 -5.363039,-9.224 -5.217937,-9.36274 0.07536,-0.072 1.975356,0.76513 4.222347,1.86029 2.246978,1.09517 4.189093,1.92995 4.315801,1.85507 0.12671,-0.0749 0.230382,-1.6268 0.230382,-3.44872 v -3.14175 l -10.921892,-5.3093 c -5.96787,-2.90107 -10.987987,-5.50859 -11.068742,-5.50859 -0.201555,0 -0.122898,5.60548 0.09342,6.58782 0.150769,0.68461 0.54642,0.97728 2.877432,2.12844 1.486444,0.73408 2.931674,1.56752 3.211621,1.85209 0.279945,0.28456 2.166705,3.37489 4.19281,6.86739 8.227951,14.18296 11.031192,18.97656 11.181075,19.11987 0.0866,0.0828 0.157528,-1.65625 0.157528,-3.86463 z"
       id="path1164-3-7"
       sodipodi:nodetypes="ccssssscsscsssscc" />
    <path
       style="fill:url(#linearGradient10728);fill-opacity:1;stroke-width:0.24754"
       d="m -84.548428,124.77295 c -6.224323,-2.99148 -11.316959,-5.49072 -11.316959,-5.55388 0,-0.0632 0.419861,-0.3212 0.933039,-0.57345 l 0.933029,-0.45862 3.429373,1.57566 c 8.520165,3.91473 13.162191,5.21548 16.845345,4.7203 1.962386,-0.26383 6.710814,-2.22473 7.43625,-3.07085 0.402848,-0.46987 0.449883,-1.54076 0.149881,-3.41265 l -0.194614,-1.21436 0.921799,-0.42117 c 1.472894,-0.67295 2.216997,-0.53199 3.78334,0.71668 0.669167,0.53345 0.871581,0.56779 2.46182,0.41766 2.259215,-0.21329 2.872887,0.0561 4.204893,1.84611 l 1.073506,1.44259 -3.763985,1.80617 c -2.070183,0.99339 -6.400594,3.11997 -9.623129,4.72574 -3.222543,1.60577 -5.881089,2.87451 -5.90789,2.86724 -0.02676,-0.007 -5.141366,-2.42168 -11.365698,-5.41317 z m -1.327686,-5.55608 c -0.834542,-0.16645 -2.41335,-0.71711 -3.508448,-1.2237 l -1.991099,-0.92106 1.145909,-0.55798 c 1.020564,-0.49694 1.332733,-0.53643 2.853867,-0.36096 2.18642,0.25219 3.59579,-0.008 5.798956,-1.06969 2.351699,-1.13349 3.261302,-1.6085 3.414051,-1.78285 0.301025,-0.34356 2.165096,-1.02027 3.176253,-1.15306 1.508433,-0.19809 3.074309,0.22762 4.964846,1.34982 0.877411,0.52081 2.254837,1.08447 3.060963,1.25259 1.649385,0.34398 1.987782,0.63108 1.306448,1.10841 -0.315391,0.22096 -1.26118,0.32669 -2.922351,0.32669 -2.018941,0 -2.85421,0.1241 -4.693401,0.69735 -1.230558,0.38355 -2.806393,0.86949 -3.501845,1.07989 -0.695452,0.2104 -2.0378,0.65983 -2.982997,0.99875 -1.914184,0.68636 -3.607604,0.75713 -6.121152,0.2558 z"
       id="path1588-6"
       sodipodi:nodetypes="ssscsssscsssscsscssscsscssssssssss" />
    <path
       style="fill:#18495d;fill-opacity:1;stroke-width:0.264583"
       d="m -81.665182,120.21491 c -1.74689,-0.89739 -1.97435,-1.12698 -1.98032,-1.99885 -0.008,-1.15582 2.94816,-2.19452 8.49334,-2.98432 3.366402,-0.47948 6.847882,0.39032 5.780662,1.44422 -0.21309,0.21042 -2.19394,1.28199 -4.401902,2.38125 -4.70828,2.34409 -5.38817,2.44382 -7.89178,1.1577 z"
       id="path2612-8" />
  </g>
</svg>
</div>
"""

# Display the animated SVG
st.markdown(animated_svg, unsafe_allow_html=True)

st.markdown('<br /><br /><br /><br /><br />', unsafe_allow_html=True)
new_title = """
<p
    style="font-family:sans-serif;text-align: center;
    color:Gray;
    font-size: 18px;
    ">I am TensorTractLab. What would you like me to do?
</p>
"""
st.markdown(new_title, unsafe_allow_html=True)

st.markdown("""
            <style>
                div[data-testid="stColumn"] {
                     margin: auto;
                    width: fit-content !important;
                    flex: unset;
                }
                div[data-testid="stColumn"] * {
                    width: fit-content !important;
                }
            </style>
            """, unsafe_allow_html=True)
#col1, col2, col3 = st.columns(3)
#col1, col2 = st.columns(2)

#with col1:
#   if st.button("Text-to-Articulators"):
#      st.switch_page("pages/text-to-articulators.py")
#
#
#with col2:
#   if st.button("Speech-to-Articulators"):
#        st.switch_page("pages/speech-to-articulators.py")

#with col3:
#   if st.button("Voice-Conversion"):
#      st.switch_page("pages/Voice-Conversion.py")

#with col4:
#   if st.button("Advanced Speech Editing"):
#      # link to the other page
#      st.write("Resynthesize Audio")

# First button centered
col1, col2, col3 = st.columns([3, 1, 3])  # center column is narrow
with col2:
   st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
   if st.button("Speech-to-Articulators"):
        st.switch_page("pages/speech-to-articulators.py")
   st.markdown("</div>", unsafe_allow_html=True)

# Spacing between buttons
st.markdown("<div style='height: 1em'></div>", unsafe_allow_html=True)

# Second button centered
col1, col2, col3 = st.columns([3, 1, 3])
with col2:
   st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
   if st.button("Text-to-Articulators"):
      st.switch_page("pages/text-to-articulators.py")
   st.markdown("</div>", unsafe_allow_html=True)




# use session state to store the model to access it in different pages
if 'ttl' not in st.session_state:
    st.session_state.ttl = load_model()