import pandas as pd
import numpy as np
import os

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


def transform_df(df: pd.DataFrame) -> (pd.DataFrame):
    '''Accepts a pd.DataFrame of survey results as input, applies a
    scoring system to the responses, fills nans with the median score, 
    and renames the columns for brevity. Returns the modified pd.DataFrame
    as output.
    '''
    df.replace(("Strongly Disagree", "Disagree", 
            "Slightly Disagree", "Unsure/No Opinion",
            "Slightly Agree", "Agree", "Strongly Agree"),
            (-3,-2,-1,0,1,2,3), inplace=True)
    df.replace(("Yes", "No"), (1,0), inplace=True)
    df = df.dropna(thresh=len(df) * .9, axis=1)
    df = df.fillna(df.median())
    df.columns = df.columns.str.replace('For each of the statements below, please indicate how strongly you agree or disagree.', "", regex=True)
    return df


def gen_col_idx(df: pd.DataFrame) -> (list, list, list, list, list):
    '''A helper function to identify the necessary columns in the
    pd.DataFrame of survey results and return the column indexs for 
    the groups of columns indicated by the variables for the topic groups.
    It returns column indexs, and a variable for each topic containing 
    the names of the columns.'''
    col_idx = list(zip(list(df.columns), range(len(df.columns))))

    ia_cols = ['HEALTH CARE:  [I support single-payer, universal health care]',
          'RELIGION:  [My religious values should be spread as much as possible]',
          'EDUCATION:  [Education is an individual’s investment in themselves, therefore they should bare the cost, not the taxpayer]']

    sl_cols = ['THE ENVIRONMENT:  [Stricter environmental laws and regulations cost too many jobs and hurt the economy]',
              'THE ENVIRONMENT:  [Climate change is currently one of the greatest threats to our way of life]',
              'PRIVACY:  [The sacrifice of some civil liberties is necessary to protect us from acts of terrorism]',
              'DOMESTIC POLICY:  [Charity is better than social security programs as a means of helping the genuinely disadvantaged]',
              'THE FUTURE:  [It is important that we maintain the traditions of our past]']

    cn_cols = ['RACE, GENDER, SEXUAL ORIENTATION:  [Our country has made the changes needed to give minorities equal rights]',
              'JOBS & THE ECONOMY:  [Those who are able to work, and refuse the opportunity, should not expect society’s support]',
              'HEALTH CARE:  [Physician-assisted suicide should be legal]',
              'THE FUTURE:  [It is important that we think in the long term, beyond our life spans]']

    rb_cols = ['JOBS & THE ECONOMY:  [The government should do more to help needy citizens, even if it means going deeper into debt]',
              'RELIGION:  [My religious values should be spread as much as possible]',
               'RACE, GENDER, SEXUAL ORIENTATION:  [Abortion, when the woman’s life is not threatened, should always be illegal]']
    
    return col_idx, ia_cols, sl_cols, cn_cols, rb_cols


def get_result_key() -> (dict):
    '''A helper function that generates a key to interpret the results of the model.'''
    result_key = {}
    result_key['ia'] = {}
    result_key['ia'][0] = {}
    result_key['ia'][1] = {}
    result_key['ia'][0]['name'] = 'Institutionalist'
    result_key['ia'][1]['name'] = 'Anarchist'
    result_key['ia'][0]['letter'] = 'I'
    result_key['ia'][1]['letter'] = 'A'
    result_key['ia'][0]['blurb'] = 'The Institutionalist typically believes that our current system suits us well.\
                                    They may be traditionalists, believing that our past or current ways of living \
                                    are superior to what the Anarchist may be proposing for the future.\
                                    They are inherently trusting of our public and private sector leadership and \
                                    believe that they have our best interests at heart.'
    result_key['ia'][1]['blurb'] = 'The Anarchist believes that our system is broken and needs radical, systemic \
                                    changes. They are overwhelmingly for single payer universal health care and \
                                    alleviating the cost of college education. \
                                    They are distrustful of authority, be it in the public or private sector, and believe \
                                    that no single person gets to own the Truth.'
    result_key['sl'] = {}
    result_key['sl'][0] = {}
    result_key['sl'][1] = {}
    result_key['sl'][0]['name'] = 'Short-Term Goal Orientation'
    result_key['sl'][1]['name'] = 'Long-Term Goal Orientation'
    result_key['sl'][0]['letter'] = 'S'
    result_key['sl'][1]['letter'] = 'L'
    result_key['sl'][0]['blurb'] = 'People with Short-Term Goal Orientation still think about the future, but they also \
                                    believe “we cannot deal with the challenges of tomorrow until we deal with the \
                                    challenges of today.\" \
                                    They tend to be more against invasions of privacy done in the \
                                    name of long-term security, and climate change is important, but not if it costs \
                                    us jobs today. They are also less likely to vote than their more \
                                    Long-Term counterparts.'
    result_key['sl'][1]['blurb'] = 'Folks who have Long-Term Goal Orientation know that some of our hardest \
                                    challenges require a long-game approach. They understand \
                                    the need for regulations to prevent climate change,even if it means \
                                    hurting the economy in the short term; they believe that climate change \
                                    is one of our greatest threats. \
                                    They are also more likely to vote than their more Short-Term counterparts.'

    result_key['cn'] = {}
    result_key['cn'][0] = {}
    result_key['cn'][1] = {}
    result_key['cn'][0]['name'] = 'Collectivist'
    result_key['cn'][1]['name'] = 'Individualist'
    result_key['cn'][0]['letter'] = 'C'
    result_key['cn'][1]['letter'] = 'N'
    result_key['cn'][0]['blurb'] = 'The Collectivist believes that citizenship is a social contract: \
                                    that they should give to their community, and in turn the community \
                                    should give back when they are in need. They believe that society has not \
                                    done enough to ensure equal rights for racial and ethnic minorities. \
                                    They are also highly likely to support personal rights of others to make their \
                                    own health decisions, such as physician-assisted suicide.'
    result_key['cn'][1]['blurb'] = 'Individualists are likely to believe in personal responsibility, \
                                    and the idea that for one person to win, another person must lose. \
                                    They are more likely to believe that society has evened the playing field \
                                    appropriately, and therefore everyone now must hold themselves accountable \
                                    and get to work. Unlike their Collectivist counterparts, \
                                    an Individualist is far more likely to believe that the best way for anyone \
                                    to change their life is to "pull themselves up by their bootstraps."'

    

    result_key['rb'] = {}
    result_key['rb'][0] = {}
    result_key['rb'][1] = {}
    result_key['rb'][0]['name'] = 'Moral Relativist'
    result_key['rb'][1]['name'] = 'Moral Absolutist'
    result_key['rb'][0]['letter'] = 'R'
    result_key['rb'][1]['letter'] = 'B'
    result_key['rb'][0]['blurb'] = 'The Relativist believes that, when it comes to life, there is always \
                                    a grey area. They believe “what works for me may not work for someone else” \
                                    and that tolerance is a sacred guiding principle of community building. \
                                    Whether devoutly religious and pro-life, or atheist and pro-choice, \
                                    each would respect the other\'s right to believe what they do.'
    result_key['rb'][1]['blurb'] = 'The Absolutist believes that life is black and white, and there is a \
                                    right and wrong way in all things. They are more likely to believe that \
                                    the ends justify the means and are more likely to disagree with the notion \
                                    that it is the government\'s responsibility to help the needy. An Absolutist \
                                    is less likely to change their perspective when presented with a new point of \
                                    view and may openly disagree with others\' choices.'

    return result_key


def result_files(ia: int, sl: int, ci: int, ar:int, 
                 result_key: dict) -> (pd.DataFrame):
    '''Accepts a result as an integer for each topic and the
    result key. The data for each result is retrieved from the result
    and creates a pd.DataFrame to store the corresponding values for
    each result. Returns the pd.DataFrame of results.
    '''
    results_df = pd.DataFrame([[ia], [sl], [ci], [ar]]).T
    results_df = results_df.rename(columns={0: 'ia', 1: 'sl',
                                            2: 'cn', 3: 'rb'})
    for col in results_df.columns:
        val = results_df[col].iloc[0]
        name = result_key[col][int(val)]['name']
        letter = result_key[col][int(val)]['letter']
        blurb = result_key[col][int(val)]['blurb']
        
        results_df[str(col) + '_name'] = name
        results_df[str(col) + '_letter'] = letter
        results_df[str(col) + '_blurb'] = blurb
        
    return results_df


def gen_topic_idx(df: pd.DataFrame, topics: list, 
                  col_idx: list) -> (pd.DataFrame, pd.DataFrame,
                                     pd.DataFrame, pd.DataFrame):
    '''Accepts a dataframe of X data, a list of columns
    for X topics, and a column index list. For each topic
    a column index of relevant columns in the dataframe
    is retrieved. Four new dataframe are returned, one for
    each set of topics.
    '''
    column_idxs = []
    for topic in topics:
        coldex = []
        for top in topic:
            idx = [t[1] for t in col_idx if t[0] == top]
            coldex += idx
        column_idxs.append(coldex)
    ia_df = df.iloc[:,column_idxs[0]]
    sl_df = df.iloc[:,column_idxs[1]]
    cn_df = df.iloc[:,column_idxs[2]]
    rb_df = df.iloc[:,column_idxs[3]]
    
    return ia_df, sl_df, cn_df, rb_df


def train_kmeans(X: pd.DataFrame, x: pd.DataFrame) -> (int): 
    '''Accepts X data for a given topic, applies StandardScaler
    and uses it to train a K-Means model. That model is then used 
    to predict the results data provided by x.
    '''
    X = X.fillna(0).to_numpy()
    x = x.fillna(0).to_numpy().reshape(1, -1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    x_scaled = scaler.fit_transform(x)
    kmeans = KMeans(n_clusters=2, init="k-means++", n_init=50, 
                max_iter=500, random_state=10)
    kmeans = kmeans.fit(X_scaled)
    return kmeans.predict(x)[0]


def get_recs() -> (pd.DataFrame):
    '''Initiates a call to the Google Sheets API, by retrieving and
    authenticating credentials and retrieving the spreadsheet for the
    survey. Returns a pd.DataFrame of the survey.
    '''
    
    scope = ['https://spreadsheets.google.com/feeds',
             'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
    client = gspread.authorize(creds)
    sheet = client.open('Politics of Earth Research Survey (Responses)')
    sheet_instance = sheet.get_worksheet(0)
    records_data = sheet_instance.get_all_records()
    records_df = pd.DataFrame.from_dict(records_data)

    return records_df


def predict_profile() -> (list, list, list):
    '''Initiates a group of functions that retrieve, transform, divide
    and filter the survey data. The result is three lists of values
    that are passed along to create the results cards for a profile.
    '''
    data = get_recs()
    data = transform_df(data)
    col_idx, ia_cols, sl_cols, cn_cols, rb_cols = gen_col_idx(data)
    topics = [ia_cols, sl_cols, cn_cols, rb_cols]
    ia_df, sl_df, cn_df, rb_df = gen_topic_idx(data, topics, col_idx)
    
    ia_X = ia_df.iloc[:-1]
    sl_X = sl_df.iloc[:-1]
    cn_X = cn_df.iloc[:-1]
    rb_X = rb_df.iloc[:-1]
    
    ia_df = ia_df.drop(ia_df.loc[ia_df.values == ''].index)
    sl_df = sl_df.drop(sl_df.loc[sl_df.values == ''].index)
    cn_df = cn_df.drop(cn_df.loc[cn_df.values == ''].index)
    rb_df = rb_df.drop(rb_df.loc[rb_df.values == ''].index)
    
    ia_x = ia_df.iloc[-1:]
    sl_x = sl_df.iloc[-1:]
    cn_x = cn_df.iloc[-1:]
    rb_x = rb_df.iloc[-1:]
    
    ia_y = train_kmeans(ia_df, ia_x)
    sl_y = train_kmeans(sl_df, sl_x)
    cn_y = train_kmeans(cn_df, cn_x)
    rb_y = train_kmeans(rb_df, rb_x)
    
    rk = get_result_key()
    results_data = result_files(ia_y, sl_y, cn_y, rb_y, rk)
    letters = [results_data['ia_letter'][0], results_data['sl_letter'][0], 
           results_data['cn_letter'][0], results_data['rb_letter'][0]]
    names = [results_data['ia_name'][0], results_data['sl_name'][0], 
           results_data['cn_name'][0], results_data['rb_name'][0]]
    blurbs = [results_data['ia_blurb'][0], results_data['sl_blurb'][0], 
           results_data['cn_blurb'][0], results_data['rb_blurb'][0]]
    
    return letters, names, blurbs


def make_card_one(letters: list) -> (dbc.Card):
    '''A helper function that accepts a list of letters as input and
    generates a card with a unique profile based on the letters.
    '''
    profile = str(' '.join(letters))
    profile_blurb = '''Click the tabs to learn about the different 
                     facets of your political profile.'''
    return dbc.Card(
            [dbc.CardBody(
                [html.H5("You're a..."),
                 html.H1(profile),
                 html.P(profile_blurb)
                 ])])


def make_card(name: str, blurb: str) -> (dbc.Card):
    '''A helper function that accepts a name and blurb to generate a
    a descriptive card for that name and blurb.
    '''
    return dbc.Card(
            [dbc.CardBody(
                [html.H5("You're a(n)..."),
                 html.H1(name),
                 html.P(blurb)
                 ])])
