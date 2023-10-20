import pandas as pd
import dash
from dash import dcc
from dash import html, register_page
import dash_bootstrap_components as dbc
from dash import Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap

nurse = pd.read_excel('assets/간호법_연도별_대분류_기사개수.xlsx')
nurse = nurse[['year', 'big_cls', 'count']]

nurse['x'] = [-2, -1.6, 0.4, 0.5, 1.8,
              -2, -1.6, 0.4, 0.5, 1.8,
              -2, -1.6, 0.4, 0.5, 1.8,
              -6, -3, 0.8, 5, 4,
              -6, -3, 0.8, 5, 4]
nurse['y'] = [-1.4, 1.5, -2.5, 2, -1,
              -1.4, 1.5, -2.5, 2, -1,
              -1.4, 1.5, -2.5, 2, -1,
              -5, 4, -7, 7, -2,
              -5, 4, -7, 7, -2]
nurse['big_cls'] = ['법안배경', '여야정쟁', '법안상충', '의견표출', '사설']*5
nurse['idx'] = nurse.index
nurse['year_num'] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]

drop_idx = nurse[nurse['count']==0].index
nurse = nurse.drop(drop_idx)


layout_nurse = html.Div([
dbc.Container([html.Br(), html.Br(), html.Br(), html.Br()], fluid=True),

    dbc.Container([
        dbc.Row(
        [dbc.Col(html.Img(id='nurse-temp', style={"height":"450px", "padding-right":20, "margim-right":30}), md=2),

        dbc.Col([dcc.Graph(id='nurse-clust'),
         html.Div([dcc.Slider(0, 4, value=0, step=1,
                              marks={0:{'label':'q1'}, 1:{'label':'q2'}, 2:{'label':'q3'}, 3:{'label':'q4'}, 4:{'label':'q5'}},
                              id = 'nurse-slider')
                  ],style={"marginTop":"20px"})],
                 style={"padding-left":1, "padding-right":10, "padding-bottom":10, "padding-top":2, "marginTop":"5px"}
                 , md=4),
         dbc.Col([
             dbc.Row([dbc.Col([dcc.Graph(id='nurse-donut', style={"height":"150px"})]),
                      dbc.Col(html.Img(id='nurse-words', style={"height":"150px", "width":"300px"}))],
                     style={'height':'30%'}),
            dbc.Row(id='nurse-cards', style={'height':'50%', 'margin-top':15, 'margin-bottom':15, 'margin-left':15, 'margin-right':1})
            ],
                 style={"padding-left":10, "padding-right":10, "padding-bottom":10, "padding-top":2, "margin-top":"5px"},
                 md=6),
         ], style={"height":"500px","padding-left":1, "padding-right":10, "padding-bottom":10, "padding-top":10, "margin-left":1})
    ],fluid=True),
],)

@callback(Output('nurse-donut', 'figure'), Input('nurse-slider', 'value'))
def nurse_update_donut(selected_qt):
    if selected_qt == 0 :
        data = [4229, 1526, 4239]  # 예시 데이터  # 예시 레이블
    elif selected_qt == 1 :
        data = [3253, 633, 1784]
    elif selected_qt == 2 :
        data = [324, 631, 1520]
    elif selected_qt == 3 :
        data = [121, 50, 52]
    else:
        data = [4329, 4271, 4341]

    fig = go.Figure(go.Pie(values=data, labels=["여당", "야당", "여/야 or 그 외"], hole=0.3,
                           textinfo='label+percent', textposition='inside'))
    fig.update_layout({'margin': {'l': 10, 'b': 1, 't': 1, 'r': 10}})
    fig.update_layout(showlegend=False)
    return fig

@callback(Output('nurse-clust', 'figure'),Input('nurse-slider', 'value'))
def nurse_update_clust(selected_qt):
    if selected_qt in [0, 1, 2]:
        data = nurse[nurse['year_num']==selected_qt]
        fig = go.Figure(go.Scatter(x=data['x'], y=data['y'], mode='markers+text', customdata=data['idx'],
                                   text=data['big_cls'], textposition="top center", textfont=dict(size=15),
                                   marker=dict(size=data['count'], sizemin=10,
                                   color=['#2772db', '#2eb872', '#facf5a', '#ff5959', '#2772db']),
                                   ))

        fig.update_layout(go.Layout(paper_bgcolor='#ededed', plot_bgcolor='#ededed'))

        fig.update_xaxes(range=[-4, 4], title=None, showticklabels=False, showgrid=False)
        fig.update_yaxes(range=[-4, 4], title=None, showticklabels=False, showgrid=False)
        fig.update_layout({'margin': {'l': 10, 'b': 10, 't': 10, 'r': 10}})
        fig.update_coloraxes(showscale=False)
        fig.update_layout(showlegend=False)
        fig.update_layout(clickmode='event+select')
        return fig

    elif selected_qt == 3:
        data = nurse[nurse['year_num'] == selected_qt]
        fig = go.Figure(go.Scatter(x=data['x'], y=data['y'], mode='markers+text', customdata=data['idx'],
                                   text=data['big_cls'], textposition="middle center", textfont=dict(size=18),
                                   marker=dict(size=[15, 250, 52, 150, 250],
                                               color=['#2772db', '#2eb872', '#facf5a', '#ff5959', '#2772db']),
                                   ))

        fig.update_layout(go.Layout(paper_bgcolor='#ededed', plot_bgcolor='#ededed'))

        fig.update_xaxes(range=[-10, 10], title=None, showticklabels=False, showgrid=False)
        fig.update_yaxes(range=[-10, 10], title=None, showticklabels=False, showgrid=False)
        fig.update_layout({'margin': {'l': 10, 'b': 10, 't': 10, 'r': 10}})
        fig.update_coloraxes(showscale=False)
        fig.update_layout(showlegend=False)
        fig.update_layout(clickmode='event+select')
        return fig

    else:
        data = nurse[nurse['year_num'] == selected_qt]
        fig = go.Figure(go.Scatter(x=data['x'], y=data['y'], mode='markers+text', customdata=data['idx'],
                                   text=data['big_cls'], textposition="middle center", textfont=dict(size=18),
                                   marker=dict(size=[47, 200, 29, 80, 170],
                                               color=['#2772db', '#2eb872', '#facf5a', '#ff5959', '#2772db']),
                                   ))

        fig.update_layout(go.Layout(paper_bgcolor='#ededed', plot_bgcolor='#ededed'))

        fig.update_xaxes(range=[-10, 10], title=None, showticklabels=False, showgrid=False)
        fig.update_yaxes(range=[-10, 10], title=None, showticklabels=False, showgrid=False)
        fig.update_layout({'margin': {'l': 10, 'b': 10, 't': 10, 'r': 10}})
        fig.update_coloraxes(showscale=False)
        fig.update_layout(showlegend=False)
        fig.update_layout(clickmode='event+select')
        return fig

@callback(Output('nurse-cards', 'children'),Input('nurse-slider', 'value'))
def nurse_update_cards(selected_qt):
    if selected_qt == 0:
        cards = dbc.Card([
            dbc.CardBody(
                 [html.P([dbc.Button(['간호법'], n_clicks=0, disabled=True, size='md',
                                        style={"background-color": "#3876BF", "outline": False, "border-radius": "10px"}), ' ',
                          dbc.Button(['간호사'], n_clicks=0, disabled=True, size='md',
                                     style={"background-color": "#3876BF", "outline": False, "border-radius": "10px"}), ' ',
                          dbc.Button(['정부'], n_clicks=0, disabled=True, size='md',
                                     style={"background-color": "#3876BF", "outline": False, "border-radius": "10px"}),
                          ]),
                  html.Hr(),
                  html.H4('핵심 요약문'),
                html.P('간호법 제정에 대한 논의가 활발히 진행되고 있는 가운데, 정치권에서는 이에 대한 다양한 반응이 나타나고 있다. 특히 서울시장 보궐선거 예비 후보인 오신환은 경쟁자인 나경원 후보의 관련 공약을 비판하는 한편, 경북도립 안동의료원에서는 간호사들에게 특정 정당 가입과 후원을 강요하는 사건이 발생하여 사회적 논란이 일었다. 한편 대한간호협회는 간호법 제정을 통해 간호사의 역할을 강화하고 보건 의료체계를 개혁하겠다는 의지를 밝혔다.'),
        ], style={"margin":10, "padding":5})])
        return cards

    elif selected_qt == 1:
        cards = dbc.Card([
                dbc.CardBody(
                    [html.P([dbc.Button(['간호법 제정'], n_clicks=0, disabled=True, size='md',
                                        style={"background-color": "#3876BF", "outline": False, "border-radius": "10px"}), ' ',
                          dbc.Button(['간호사의 역할 및 권한'], n_clicks=0, disabled=True, size='md',
                                     style={"background-color": "#3876BF", "outline": False, "border-radius": "10px"}), ' ',
                          dbc.Button(['정치적 고려'], n_clicks=0, disabled=True, size='md',
                                     style={"background-color": "#3876BF", "outline": False, "border-radius": "10px"}),]
                        ),
                     html.Hr(),
                     html.H4('핵심 요약문'),
                     html.P(
                         '각 기사는 간호법 제정과 간호사의 역할에 대한 중요성을 강조하고 있습니다. 안동의료원에서는 특정 정당 가입과 후원을 강요하는 사건이 발생하여 경찰에 고발되었으며, 더불어민주당 대표인 송영길은 여야정 협의체를 통해 백신, 손실보상 등 협의할 문제가 많다며 여야간 대화를 촉구하였습니다. 또한, 대한간호협회에서는 코로나19 팬데믹 상황에서 의료인력 확보의 중요성을 짚었으며, 불법진료와 관련된 문제도 지적하였습니다.이들은 모두 간호법 제정을 통해 간호사들의 역할과 권한을 명확히 하고, 그들이 직면하는 어려움을 해결해 나가야 한다는 점에서 일치하는 입장을 보이고 있습니다.'
                     )],
                )
            ], style={"margin": 10})
        return cards

    elif selected_qt == 2:
        cards = dbc.Card([
            dbc.CardBody(
                [html.P([dbc.Button(['간호법 제정'], n_clicks=0, disabled=True, size='sm',
                                        style={"background-color": "#3876BF", "outline": False, "border-radius": "10px"}), ' ',
                          dbc.Button(['의사 협회의 반발'], n_clicks=0, disabled=True, size='sm',
                                     style={"background-color": "#3876BF", "outline": False, "border-radius": "10px"}), ' ',
                          dbc.Button(['국회 보건복지 위원회'], n_clicks=0, disabled=True, size='sm',
                                     style={"background-color": "#3876BF", "outline": False, "border-radius": "10px"}),]
                    ),
                 html.Hr(),
                 html.H4('핵심 요약문'),
                 html.P(
                     '국회 보건복지위원회가 간호법을 의결하였으나, 전국 시도의사회와 대한의사협회는 이에 강력하게 반발하고 있습니다. 의료 관련 단체들은 국회가 특정 직업군만을 위한 법안 제정에 나서는 것을 문제삼으며, 이에 대해 총력 투쟁을 선언하였습니다. 한편 더불어민주당은 5월 중 본회의에서 간호법을 상정할 계획이며, 국민의힘은 법안소위 소집에 유감을 표해 별도 법안인 간호법 자체를 반대하는 입장입니다.'
                 )],
            )
        ], style={"margin": 10})
        return cards

    elif selected_qt == 3:
        cards = dbc.Card([
            dbc.CardBody(
                [html.P([dbc.Button(['간호법 제정'], n_clicks=0, disabled=True, size='md',
                                        style={"background-color": "#3876BF", "outline": False, "border-radius": "10px"}), ' ',
                          dbc.Button(['국민의힘의 반발'], n_clicks=0, disabled=True, size='md',
                                     style={"background-color": "#3876BF", "outline": False, "border-radius": "10px"}), ' ',
                          dbc.Button(['더불어민주당'], n_clicks=0, disabled=True, size='md',
                                     style={"background-color": "#3876BF", "outline": False, "border-radius": "10px"}),]
                    ),
                 html.Hr(),
                 html.H4('핵심 요약문'),
                 html.P(
                     '더불어민주당은 국회 보건복지위원회에서 간호법을 의결하였으며, 5월 본회의에서 통과할 것으로 예상되고 있습니다. 한편, 국민의힘은 이에 대해 비판적인 입장을 보이며, 의료계도 극단적인 대립 양상을 보이고 있다고 지적하였습니다. 민주당은 이러한 상황에도 불구하고 간호법과 의료법 등 민생법안 처리를 약속하였습니다. 그러나 양곡관리법 개정안은 최종 부결되었으며, 이에 대해 민주당은 후속 입법을 통해 반드시 양곡관리법을 정상화할 것이라고 밝혔습니다.'
                 )],
            )
        ], style={"margin": 10})
        return cards

    else:
        cards = dbc.Card([
            dbc.CardBody(
                [html.P([dbc.Button(['간호법 제정안'], n_clicks=0, disabled=True, size='md',
                                        style={"background-color": "#3876BF", "outline": False, "border-radius": "10px"}), ' ',
                          dbc.Button(['제의요구권(거부권)'], n_clicks=0, disabled=True, size='md',
                                     style={"background-color": "#3876BF", "outline": False, "border-radius": "10px"}), ' ',
                          dbc.Button(['국민의 힘'], n_clicks=0, disabled=True, size='md',
                                     style={"background-color": "#3876BF", "outline": False, "border-radius": "10px"}),]),
                 html.Hr(),
                 html.H4('핵심 요약문'),
                 html.P(
                     '윤석열 대통령이 간호법 제정안에 대해 재의요구권을 행사한 것에 대해 국민의힘은 "불가피한 선택"이라고 밝혔으며, 이 법이 시행된다면 의료 협업 시스템을 붕괴시키고 국민 건강에 악영향을 끼칠 것이라는 주장과 함께, 이 법은 단독으로 더불어민주당이 밀어붙인 것으로 지적하였습니다. 그러나 여야는 간호법 재표결을 두고 맞서며 입장 차를 보였습니다. 국민의힘이 반대 입장을 고수하는 가운데, 더불어민주당은 간호법 찬성 입장에서 변하지 않으며 그 필요성과 중요성을 주장하였습니다. 결국 윤석열 대통령이 거부권을 행사한 후 본회의에서 진행된 간호법 제정안은 부결되었습니다. 이에 따라 더불어민주당은 새로운 법인 준비를 예고하며 의료체계를 내실있게 강화할 수 있는 방안 모색에 나설 계활임을 밝혔습니다.'
                 )],
            )
        ], style={"margin": 10})
        return cards

@callback(Output('nurse-words', 'src'), Input('nurse-slider', 'value'))
def nurse_upadate_wd(selected_qt):
    src = selected_qt+1
    src = 'assets/img'+str(src)+'.jpg'
    return src

@callback(Output('nurse-temp', 'src'), Input('nurse-slider', 'value'))
def nurse_upadate_temp(selected_qt):
    src = selected_qt+1
    src = 'assets/nur_'+str(src)+'.png'
    return src

register_page(
    __name__,
    name='Nurse',
    top_nav=True,
    path='/nurse',
    layout=layout_nurse
)