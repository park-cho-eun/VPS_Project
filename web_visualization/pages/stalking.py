import pandas as pd
import dash
import copy
from dash import dcc
from dash import html, register_page
import dash_bootstrap_components as dbc
from dash import Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px

#스토킹 데이터 불러오기
xy = pd.read_json('assets/final_vis.json')
xy['size'] = [208, 119, 61, 220, 110, 28, 277, 128, 18, 210, 120, 7]
label_2 = pd.read_json('assets/semifinal_news_2.json') #세부 클러스터링
label = pd.read_json('assets/semifinal_news_3.json') #대분류 클러스터

label_2 = label_2[['sliced', '2dim', '작성일', 'quarter', 'label', 'details']]
label_2['label'] = [2]*label_2.shape[0]
label = label[label['label'].isin([0,1])]
label['details'] = [10]*label.shape[0]

label_new = pd.concat([label, label_2])
label_etc = label_new[label_new['details'].isin([0, 1, 2, 3])]
xy['x_new'] = [-0.5, 0.3, 1, -0.5, 0.3, 1, -0.5, 0.3, 1, -0.5, 0.3, 1]
xy['y_new'] = [0.3, -1.2, -0.5, 0.3, -1.2, -0.5, 0.3, -1.2, -0.5, 0.3, -1.2, -0.5]
xy['label'] = ['스토킹 사례', '정부 대응', '비판', '스토킹 사례', '정부 대응', '비판',
               '스토킹 사례', '정부 대응', '비판', '스토킹 사례', '정부 대응', '비판']

'''customdt = []
for i in range(1,5):
    qua = 'q'+str(i)
    label_new[label_new['quarter']==qua]'''

laws = label_new.groupby(['quarter', 'details']).count()
laws = laws.reset_index()
laws_idx = laws[laws['details']==10].index
laws = laws.drop(laws_idx)
laws = laws.reset_index()
laws['details_name'] = ['처벌수위', '경찰 대응', '후속조치/피해자보호', '기타', '반의사불벌죄', '스토킹 개념 모호',
                        '처벌수위', '경찰 대응', '후속조치/피해자보호', '기타', '반의사불벌죄', '스토킹 개념 모호',
                        '처벌수위', '경찰 대응', '후속조치/피해자보호', '기타',
                        '처벌수위', '경찰 대응', '후속조치/피해자보호', '기타']

laws_ext = pd.read_excel('assets/law_stalking.xlsx')
laws = pd.concat([laws, laws_ext], axis=1)

layout_stalk = html.Div([
    dbc.Container([html.Br(), html.Br(), html.Br(), html.Br()],),

    dbc.Container([
        dbc.Row(
            [dbc.Col(html.Img(id='stalk-temp', style={"height":"450px", "padding-right":1, "margim-right":1}), md=2),
                dbc.Col([dcc.Graph(id='histogram'),
                      html.Div([dcc.Slider(1, 4, value=1, step=1,
                                           marks={1: {'label': 'q1'}, 2: {'label': 'q2'}, 3: {'label': 'q3'},
                                                  4: {'label': 'q4'}},
                                           id='slider-1')
                                ], style={"marginTop": "20px"})], md=4,
                        style={"padding-left": 1, "padding-right":1, "padding-top":10, "padding-bottom":10, "margin-left":1}),
             dbc.Col([html.A(dbc.Row([dcc.Graph(id='donut'), ], style={'height':'50%'})),
                      html.A(dbc.Row(id='stalking_law', style={'height':'40%'}))], md=5),
             ], style={"padding": 10})

    ], fluid=True),
],)

@callback(Output('histogram', 'figure'), Input('slider-1', 'value'))
def stalking_update_clust(selected_qt):
    qt = 'q' + str(selected_qt)
    data = xy[xy['quarter'] == qt]
    fig = go.Figure(go.Scatter(x=data['x_new'], y=data['y_new'], mode='markers+text',
                               text=data['label'], textposition="middle center", textfont=dict(size=23),
                               marker=dict(size=data['size'], color=['#2772db', '#2eb872', '#facf5a', '#ff5959']),
                               ))

    fig.update_layout(go.Layout(paper_bgcolor='#ededed', plot_bgcolor='#ededed'))

    fig.update_traces(marker={'size': data['size'].tolist(), 'line': {'width': 2, 'color': 'white'}})
    fig.update_layout(showlegend=False)
    fig.update_coloraxes(showscale=False)
    fig.update_layout({'margin': {'l': 1, 'b': 10, 't': 10, 'r': 1}})
    fig.update_xaxes(range=[-2, 2], title=None, showticklabels=False, showgrid=False)
    fig.update_yaxes(range=[-2, 2], title=None, showticklabels=False, showgrid=False)
    return fig

@callback(Output('donut', 'figure'), Input('slider-1', 'value'))
def stalking_update_donut(selected_qt):
    qt = 'q' + str(selected_qt)
    data = copy.deepcopy(laws[laws['quarter'] == qt])
    fig = px.bar(data, x='sliced', y='details_name', color='details_name', orientation='h'
                 , text='details_name', custom_data='oreders')
    fig.update_layout(go.Layout(plot_bgcolor='#ededed'))
    fig.update_layout(showlegend=False)
    fig.update_coloraxes(showscale=False)
    fig.update_yaxes(title=None, showgrid=True, visible=False)
    fig.update_xaxes(title=None, showgrid=True, visible=False, range=[0, 20])
    fig.update_layout({'margin': {'l': 10, 'b': 10, 't': 10, 'r': 1}})
    fig.update_layout(clickmode='event+select')

    return fig

@callback(Output('stalking_law', 'children'), Input('donut', 'clickData'))
def youth_update_cards(clickData):
    if clickData is None:
        cards = dbc.Card([
            dbc.CardBody(
                [html.H4("해당 시기 관련 법률", className="card-title"),
                 html.Br(),
                 html.P('위 그림의 일부를 클릭해주세요.'),],
            ),
        ], style={"margin":30})

        return cards

    else:
        n = clickData['points'][0]['customdata']
        data = laws[laws['oreders']==n[0]]

        cards = dbc.Card([
            dbc.CardBody(
                [html.H4(data['law_title']),
                 html.H5(data['law_subtitle']),
                 html.P(data['law_cont'])],
            ),
        ], style={"margin":30})
        return cards

@callback(Output('stalk-temp', 'src'), Input('slider-1', 'value'))
def stalk_upadate_temp(selected_qt):
    src = 'assets/stalk_'+str(selected_qt)+'.png'
    return src

register_page(
    __name__,
    name='Stalking',
    top_nav=True,
    path='/stalking',
    layout=layout_stalk
)
