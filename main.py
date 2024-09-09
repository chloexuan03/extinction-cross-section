import joblib
import streamlit as st
import pandas as pd
import numpy as np
import time

# import plotly.express.line

st.set_page_config(
    page_title="Extinction cross section sepctroscopy",
    layout="centered",
    initial_sidebar_state="expanded",
)


def main():
    model = joblib.load("wl_ext_mod_DecisionTree.joblib")
    dictionary = joblib.load("material_dictionary.joblib")
    limit_wl = joblib.load("limit_wl.joblib")
    element, radius, bool_plot = cs_sidebar(dictionary)
    cs_body(element, radius, bool_plot, dictionary, limit_wl, model)
    return None


def cs_sidebar(dictionary):
    bool_plot = False
    st.sidebar.header("Materials' Entry")
    st.sidebar.markdown(
        ":gray[Inputs include: type of element and radius of the sphere of interest, along with the prominent range of wavelength, make a complete set of features.]"
    )
    st.sidebar.divider()
    element = st.sidebar.selectbox(
        "Choose your material",
        dictionary.keys(),
    )
    radius = st.sidebar.select_slider(
        "Slide to select the radius", options=np.arange(10.0, 100.25, 0.25)
    )
    st.sidebar.divider()
    st.sidebar.write(f"You choose {element} particle as shpere of radius {radius}(nm)")
    if st.sidebar.button("Confirmed!", use_container_width=True, type="primary"):
        bool_plot = True
        return element, radius, bool_plot
    else:
        return "...", "...", bool_plot


def run_and_plot(element, radius, dictionary, limit_wl, model):
    # Number of avaiable elements ready
    num_ele = len(dictionary)
    temp_permitivity = dictionary[element]
    permitivity_arr = [temp_permitivity] * 101
    rad_arr = [radius * 10**-3] * 101  # Normalizing parameter: micrometer -> nanometer

    # Find the convinient range of wavelength to take predict for each element
    error = True
    for i in range(num_ele):
        if limit_wl[i][0] == temp_permitivity:
            low_lim = limit_wl[i][1]
            up_lim = limit_wl[i][2]
            error = False
            break
    if error:
        print("Error")
    wl_arr = np.linspace(low_lim, up_lim, 101)

    feature = np.asarray(
        [[rad_arr[i], wl_arr[i], permitivity_arr[i]] for i in range(len(rad_arr))]
    )

    # Predict using the model specified
    q_ext = model.predict(feature)
    wl_arr_nm = wl_arr * 1000
    sigma_arr_nm = q_ext * np.pi * (radius) ** 2
    graph = pd.DataFrame(
        {"Wave Length (nm)": wl_arr_nm, "Cross Section (nm^2)": sigma_arr_nm[:, 0]}
    )
    st.line_chart(
        graph, x="Wave Length (nm)", y="Cross Section (nm^2)", color="#48CFCB"
    )


def cs_body(element, radius, bool_plot, dictionary, limit_wl, model):
    st.title("Spectroscopy of Nanoparticles")
    col1, col2, col3 = st.columns([0.6, 0.15, 0.25], vertical_alignment="center")
    col1.subheader("Extinction cross section for particle of:")
    col2.metric(label="Element", value=f"{element}", delta=None)
    col3.metric(label="Radius", value=f"{radius} nm", delta=None)
    st.write(
        "In the case where size of the scattering particles is comparable to the wavelength of the light, it is prominent to the treats particles as spherical objects. As in the Mie theory, the idea is generally used to calculate how much light is scattered."
    )
    st.write(
        "The informations of different materials are collected via https://refractiveindex.info/. We proceed to use machine learning methods to train and predict the spectroscopy of an arbitrary material. The :green-background[DecisionTreeRegressor model] has proven to be appropriate solution as it shows excellent evaluations:"
    )
    st.caption(
        ":green[ __R-squared: 0.9960, Root Mean Square Deviation: 0.0844, Mean absolute error: 0.0321__]"
    )
    text = """The data is splitted into train and test datas. In which, the feature is determined to be a dataframe of 3 components: radius, wavelength and index (which tells the type of element and is characterized by its relative permitivity). The target is then the spectroscopy of the chosen nanoparticle at any given wavelength."""

    def stream_data():
        for word in text.split(" "):
            yield word + " "
            time.sleep(0.02)

    if bool_plot:
        run_and_plot(element, radius, dictionary, limit_wl, model)
        st.write_stream(stream_data)


main()
