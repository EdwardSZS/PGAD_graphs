import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import time
import plotly.figure_factory as ff
import scipy
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import calendar
import statsmodels.tsa.api as smt
from mpl_toolkits.mplot3d import *





def main():
    page=st.sidebar.selectbox(
        "Selecciona una página",
        (
            "Inicio",
            "Descripción estadística"

        )

    )
    page_=st.session_state
    if page == "Inicio":
        inicio()

    elif page == "Descripción estadística":
        estadistica()


def inicio():
    st.title('Descripción estadística sobre le Covid_19')
    st.header("Revisón de información")
    st.subheader("Simulación de datos sobre el Covid_19 con pygame")
    ubicacion()

def estadistica():
    
    salud = st.sidebar.radio("Situaciones de salud",('Salud Buena','Salud Regular','Salud Mala'))
    
    if st.sidebar.button('Enviar consulta'):
        dibujar_graficas(salud )

def dibujar_graficas(salud ):
    st.header("Análisis de Gráficas")
    with st.spinner("Cargando...."):
         time.sleep(2)
    
    if salud == "Salud Mala":
        cargar_datos_1()
    if salud == "Salud Regular":
        cargar_datos_2()
    if salud == "Salud Buena":
        cargar_datos_3()

def tsplot2(y, y2=None, lags=None, figsize=(12, 7), style='bmh'):

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style='bmh'):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax, label='Serie 1')
        if y2 is not None:
            if not isinstance(y2, pd.Series):
                y2 = pd.Series(y2)
            y2.plot(ax=ts_ax, label='Serie 2', color='r')
        ts_ax.legend()

        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        st.pyplot(fig)
        if y2 is not None:
          p_value2 = sm.tsa.stattools.adfuller(y2)[1]
          smt.graphics.plot_acf(y2, lags=lags, ax=acf_ax)
          smt.graphics.plot_pacf(y2, lags=lags, ax=pacf_ax)
          plt.tight_layout()

def cargar_datos_1():
    st.subheader("Condición de salud mala")
    df = pd.read_csv('../Virus_02.csv',sep=',')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.set_index('Date', inplace=True)
    st.subheader("📈Evaluación de series de tiempo")  
    
    st.caption("Infectados vs Fallecidos" )
    tsplot2(df['Infected'], df['Dead'],lags=20)
    st.caption("Susceptibles vs Fallecidos" )
    tsplot2(df['Susceptible'], df['Dead'],lags=20)
    st.caption("Recuperados vs Fallecidos" )
    tsplot2(df['Recovered'], df['Dead'],lags=20)
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', fill='tozeroy', name=col))


    fig.update_yaxes(range=[0, 1000])


    fig.update_layout(
        title='Dinámica de Susceptible, Infected y Recovered',
        xaxis_title='Días',
        yaxis_title='Cantidad',
        showlegend=True
    )


    def display_hover_data(trace, points, state):
        day = points.point_inds[0]
        values = [df[col][day] for col in df.columns]
        hover_text = f'Día: {day}<br>' + '<br>'.join([f'{col}: {value}' for col, value in zip(df.columns, values)])
        fig.update_traces(hoverinfo='text', hovertext=hover_text)

    fig.for_each_trace(lambda trace: trace.on_hover(display_hover_data))


    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Dataframe"):
        st.dataframe(df)




def cargar_datos_2():
    st.subheader("Condición de salud regular")
    df = pd.read_csv('../Virus_00.csv',sep=',')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.set_index('Date', inplace=True)
    st.subheader("📈Evaluación de series de tiempo")  
    
    st.caption("Infectados vs Fallecidos" )
    tsplot2(df['Infected'], df['Dead'],lags=20)
    st.caption("Susceptibles vs Fallecidos" )
    tsplot2(df['Susceptible'], df['Dead'],lags=20)
    st.caption("Recuperados vs Fallecidos" )
    tsplot2(df['Recovered'], df['Dead'],lags=20)
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', fill='tozeroy', name=col))


    fig.update_yaxes(range=[0, 1000])


    fig.update_layout(
        title='Dinámica de Susceptible, Infected y Recovered',
        xaxis_title='Días',
        yaxis_title='Cantidad',
        showlegend=True
    )


    def display_hover_data(trace, points, state):
        day = points.point_inds[0]
        values = [df[col][day] for col in df.columns]
        hover_text = f'Día: {day}<br>' + '<br>'.join([f'{col}: {value}' for col, value in zip(df.columns, values)])
        fig.update_traces(hoverinfo='text', hovertext=hover_text)

    fig.for_each_trace(lambda trace: trace.on_hover(display_hover_data))


    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Dataframe"):
        st.dataframe(df)
    


def ubicacion():
    
    chart_data = pd.DataFrame(
       np.random.randn(1000, 2) / [15, 38] + [4.66968, -74.1121],
       columns=['lat', 'lon'])

    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=4.666968,
            longitude=-74.1121,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
               'HexagonLayer',
               data=chart_data,
               get_position='[lon, lat]',
               radius=200,
               elevation_scale=4,
               elevation_range=[0, 1000],
               pickable=True,
               extruded=True,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=chart_data,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=200,
            ),
        ],
    ))
def cargar_datos_3():
    st.subheader("Condición de salud buena")
    df = pd.read_csv('../Virus_01.csv',sep=',')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.set_index('Date', inplace=True)
    st.subheader("📈Evaluación de series de tiempo")  
    
    st.caption("Infectados vs Fallecidos" )
    tsplot2(df['Infected'], df['Dead'],lags=20)
    st.caption("Susceptibles vs Fallecidos" )
    tsplot2(df['Susceptible'], df['Dead'],lags=20)
    st.caption("Recuperados vs Fallecidos" )
    tsplot2(df['Recovered'], df['Dead'],lags=20)
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', fill='tozeroy', name=col))


    fig.update_yaxes(range=[0, 1000])


    fig.update_layout(
        title='Dinámica de Susceptible, Infected y Recovered',
        xaxis_title='Días',
        yaxis_title='Cantidad',
        showlegend=True
    )


    def display_hover_data(trace, points, state):
        day = points.point_inds[0]
        values = [df[col][day] for col in df.columns]
        hover_text = f'Día: {day}<br>' + '<br>'.join([f'{col}: {value}' for col, value in zip(df.columns, values)])
        fig.update_traces(hoverinfo='text', hovertext=hover_text)

    fig.for_each_trace(lambda trace: trace.on_hover(display_hover_data))


    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Dataframe"):
        st.dataframe(df)

if __name__== "__main__":
    main()
