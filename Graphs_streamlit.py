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
            "Descripción estadística",
            "Comparativo"

        )

    )
    page_=st.session_state
    if page == "Inicio":
        inicio()

    elif page == "Descripción estadística":
        estadistica()

    elif page ==  "Comparativo":
        comparativo()


def inicio():
    st.title('Descripción estadística sobre le Covid_19')
    st.header("Revisón de información")
    st.subheader("Simulación de datos sobre el Covid_19 con pygame")
    st.subheader("Emulación de rangos de edad para Bogotá")
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

def tsplot2(y, y2=None, lags=None, figsize=(12, 7), style='bmh',serie1="serie1",serie2="serie2"):

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style='bmh'):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax, label=serie1)
        if y2 is not None:
            if not isinstance(y2, pd.Series):
                y2 = pd.Series(y2)
            y2.plot(ax=ts_ax, label=serie2, color='r')
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
    df = pd.read_csv('Virus_02.csv',sep=',')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.set_index('Date', inplace=True)
    st.subheader("📈Evaluación de series de tiempo", divider='rainbow')  
    
    st.subheader("Infectados vs Fallecidos" , divider='rainbow')
    tsplot2(df['Infected'], df['Dead'],lags=20,serie1="Infected",serie2="Dead")
    st.subheader("Susceptibles vs Fallecidos" , divider='rainbow')
    tsplot2(df['Susceptible'], df['Dead'],lags=20,serie1="Susceptible",serie2="Dead")
    st.subheader("Recuperados vs Fallecidos" , divider='rainbow')
    tsplot2(df['Recovered'], df['Dead'],lags=20,serie1="Recovered",serie2="Dead")
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
    df = pd.read_csv('Virus_00.csv',sep=',')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.set_index('Date', inplace=True)
    st.subheader("📈Evaluación de series de tiempo", divider='rainbow')  
    
    st.subheader("Infectados vs Fallecidos" , divider='rainbow')
    tsplot2(df['Infected'], df['Dead'],lags=20,serie1="Infected",serie2="Dead")
    st.subheader("Susceptibles vs Fallecidos", divider='rainbow' )
    tsplot2(df['Susceptible'], df['Dead'],lags=20,serie1="Susceptible",serie2="Dead")
    st.subheader("Recuperados vs Fallecidos", divider='rainbow' )
    tsplot2(df['Recovered'], df['Dead'],lags=20,serie1="Recovered",serie2="Dead")
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
               radius=150,
               elevation_scale=30,
               elevation_range=[30, 90],
               pickable=True,
               extruded=True,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=chart_data,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=150,
            ),
        ],
    ))
def cargar_datos_3():
    st.subheader("Condición de salud buena")
    df = pd.read_csv('Virus_01.csv',sep=',')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df.set_index('Date', inplace=True)
    st.subheader("📈Evaluación de series de tiempo", divider='rainbow')  
    
    st.subheader("Infectados vs Fallecidos", divider='rainbow' )
    tsplot2(df['Infected'], df['Dead'],lags=20,serie1="Infected",serie2="Dead")
    st.subheader("Susceptibles vs Fallecidos", divider='rainbow' )
    tsplot2(df['Susceptible'], df['Dead'],lags=20,serie1="Susceptible",serie2="Dead")
    st.subheader("Recuperados vs Fallecidos", divider='rainbow' )
    tsplot2(df['Recovered'], df['Dead'],lags=20,serie1="Recovered",serie2="Dead")
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

def comparativo ():
#SALUD MALA
    df1 = pd.read_csv('Virus_02.csv',sep=',')
    #SALUD REGULAR
    df2 = pd.read_csv('Virus_00.csv',sep=',')
    #SALUD MALA
    df3 = pd.read_csv('Virus_01.csv',sep=',')
    data1 = calculate_derivatives(df1)
    data2 = calculate_derivatives(df2)
    data3 = calculate_derivatives(df3)
    fig = go.Figure()
    # Streamlit app
    st.title('Comparativos de velocidad y máximos entre los tres casos de salud')
    st.subheader("Comportamiento de Infectados en las distintas condiciones de salud", divider='rainbow' )
    st.caption("Caso 1: Mala salud, Caso2: Salud Regular, Caso 3: Salud Buena" )
    # Plot the time series and derivatives
    fig.add_trace(go.Scatter(x=data1['Date'], y=data1['Infected'], mode='lines', name='Infected Caso 1'))
    fig.add_trace(go.Scatter(x=data1['Date'], y=data1['derivative'], mode='lines', name='Velocity Caso 1'))

    fig.add_trace(go.Scatter(x=data2['Date'], y=data2['Infected'], mode='lines', name='Infected Caso 2'))
    fig.add_trace(go.Scatter(x=data2['Date'], y=data2['derivative'], mode='lines', name='Velocity Caso 2'))

    fig.add_trace(go.Scatter(x=data3['Date'], y=data3['Infected'], mode='lines', name='Infected Caso 3'))
    fig.add_trace(go.Scatter(x=data3['Date'], y=data3['derivative'], mode='lines', name='velocity Caso 3'))

    
    fig.update_layout(
        title='Infectados en tres condiciones',
        xaxis_title='Días',
        yaxis_title='Cantidad de población',
        showlegend=True
    )
    
    st.plotly_chart(fig)

    fig1 = go.Figure()


    st.subheader("Comportamiento de Fallecidos en las distintas condiciones de salud", divider='rainbow' )
    st.caption("Caso 1: Mala salud, Caso 2: Salud Regular, Caso 3: Salud Buena" )
    # Plot the time series and derivatives
    fig1.add_trace(go.Scatter(x=data1['Date'], y=data1['Dead'], mode='lines', name='Dead Caso 1'))
    fig1.add_trace(go.Scatter(x=data1['Date'], y=data1['derivative'], mode='lines', name='Velocity Caso 1'))

    fig1.add_trace(go.Scatter(x=data2['Date'], y=data2['Dead'], mode='lines', name='Dead  Caso 2'))
    fig1.add_trace(go.Scatter(x=data2['Date'], y=data2['derivative'], mode='lines', name='Velocity Caso 2'))

    fig1.add_trace(go.Scatter(x=data3['Date'], y=data3['Dead'], mode='lines', name='Dead Caso 3'))
    fig1.add_trace(go.Scatter(x=data3['Date'], y=data3['derivative'], mode='lines', name='velocity Caso 3'))
    fig1.update_layout(
        title='Fallecidos en tres condiciones',
        xaxis_title='Días',
        yaxis_title='Cantidad de población',
        showlegend=True
    )
    st.plotly_chart(fig1)


    st.subheader("Comparativo entre Fallecidos e Infectados", divider='rainbow' )
    st.caption("Caso 1: Mala salud, Caso2: Salud Regular, Caso 3: Salud Buena" )
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=data1['Date'], y=data1['Dead'], mode='lines', name='Dead Caso 1'))
    fig2.add_trace(go.Scatter(x=data1['Date'], y=data1['Infected'], mode='lines', name='Infected Caso 1'))

    fig2.add_trace(go.Scatter(x=data2['Date'], y=data2['Dead'], mode='lines', name='Dead Caso 2'))
    fig2.add_trace(go.Scatter(x=data2['Date'], y=data2['Infected'], mode='lines', name='Infected Caso2'))

    fig2.add_trace(go.Scatter(x=data3['Date'], y=data3['Dead'], mode='lines', name='Dead Caso 3'))
    fig2.add_trace(go.Scatter(x=data3['Date'], y=data3['Infected'], mode='lines', name='Infected Salud Buena'))
    fig2.update_layout(
        title='Comparativo entre Fallecidos e Infectados',
        xaxis_title='Días',
        yaxis_title='Cantidad de población',
        showlegend=True
    )
    st.plotly_chart(fig2)

def calculate_derivatives(data):
    data['derivative'] = data['Infected'].diff()
    return data

if __name__== "__main__":
    main()
