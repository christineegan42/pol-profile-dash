import pandas as pd
import numpy as np
import os

import os
from dotenv import load_dotenv
load_dotenv(verbose=True)

import dash
from jupyter_dash import JupyterDash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import plotly.express as px
import plotly.graph_objects as go

import gspread
from oauth2client.service_account import ServiceAccountCredentials

from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from dashboard import *

form = 'https://docs.google.com/forms/d/e/1FAIpQLScp713zzPjkdbUuNVecCNwn-U3nS2P9KX4VUcyKaJhhGQCY2A/viewform'

re_bl = "We used machine-learning to determine what factors were most influential \
        to forming your unique political profile. Take the survey and submit your responses. \
        Then, click the button below to generate your profile."

letters, names, blurbs = predict_profile()

card1 = make_card_one(letters)
card2 = make_card(names[0], blurbs[0])
card3 = make_card(names[1], blurbs[1])
card4 = make_card(names[2], blurbs[2])
card5 = make_card(names[3], blurbs[3])


result_tabs = dbc.Tabs([dbc.Tab(card1, label="Your Profile"),
                     dbc.Tab(card2, label="IA"),
                     dbc.Tab(card3, label="SL"),
                     dbc.Tab(card4, label="CN"),
                     dbc.Tab(card5, label="RB")])


fade = dbc.Fade(result_tabs, id="fade-transition", is_in=True, style={"transition": "opacity 100ms ease"})

result_blurb = html.Div(dbc.Card(
                [dbc.CardBody(
                    [html.H5("What's your political profile?", 
                             className='card-title'),
                     html.P(re_bl),
                     dbc.Button("Get My Results", 
                                id="fade-transition-button", 
                                className="mb-3", n_clicks=0),
                     fade])]))


result_card = dbc.Card([dbc.CardBody([dbc.Row([result_blurb])])])


survey_card = dbc.Card(
                [dbc.CardBody(dbc.Row([html.Embed(src=form, height=900, width=750)
                    ]))], style={'width':'auto', 'height':'auto'})


app = JupyterDash(external_stylesheets=[dbc.themes.LITERA])
server = app.server


top_cell = dbc.Col([html.H2('Citizen of Earth Dashboard')], width=12)
right_col = dbc.Col([survey_card], width=6)
left_col = dbc.Col([result_card], width=6)
app.layout = html.Div([dbc.Row([top_cell]),
                        dbc.Row([left_col, right_col])])


@app.callback(
    Output("fade-transition", "is_in"),
    [Input("fade-transition-button", "n_clicks")],
    [State("fade-transition", "is_in")],
)
def toggle_fade(n_clicks, is_in):
    if n_clicks != 0:
        return fade
    else:
        return None

    
if __name__ == '__main__':
    app.run_server(debug=True)


